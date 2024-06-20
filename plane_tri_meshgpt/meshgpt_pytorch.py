from functools import partial

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Tuple, Callable, List, Dict, Any, Optional, Union
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from torch_geometric.nn.conv import SAGEConv

from einops import reduce, rearrange, pack, unpack

from einx   import get_at

from plane_tri_meshgpt.data import derive_face_edges_from_faces


def exists(v):
    return v is not None

@beartype
def discretize(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)


class MeshAutoencoder(Module, PyTorchModelHubMixin):
    @beartype
    def __init__(
            self,
            num_discrete_coors = 128,
            coor_continuous_range:Tuple[float,float] = (-1.,1.),
            dim_coor_embed = 64,
            num_discrete_confidence = 128,
            dim_confidence_embed = 16,
            encoder_dims_through_depth:Tuple[int, ...] = (
                    64, 128, 256, 256, 576
            ),
            sageconv_kwargs:dict = dict(
                normalize = True,
                project = True
            ),
            commit_loss_weight = 0.1,
            pad_id = -1,
            dim_codebook = 192,
    ):
        super().__init__()

        self.num_vertices_per_face = 3
        total_coordinates_per_face = self.num_vertices_per_face * 3

        # main face coordinate embedding
        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        # feature embedding
        self.discretize_confidence = partial(discretize,num_discrete = num_discrete_confidence, continuous_range = (0.,100))
        self.confidence_embed = nn.Embedding(num_discrete_confidence, dim_confidence_embed)

        # initial sage conv
        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        curr_dim = init_encoder_dim
        self.init_sage_conv = SAGEConv(dim_codebook,init_encoder_dim,**sageconv_kwargs)
        self.init_encoder_act_and_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(init_encoder_dim)
        )
        self.encoders = ModuleList([])
        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                curr_dim,
                dim_layer,
                **sageconv_kwargs
            )
            self.encoders.append(sage_conv)
            curr_dim = dim_layer

        self.pad_id = pad_id
        # loss related
        self.commit_loss_weight = commit_loss_weight

    # TODO:待修改，接纳face_feature
    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 'nvf', int],
        face_edges:       TensorType['b', 'e', 2, int],
        face_mask:        TensorType['b', 'nf', bool],
        face_edges_mask:  TensorType['b', 'e', bool],
        return_face_coordinates = False
    ):
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        nvf - number of vertices per face (3 or 4) - triangles vs quads
        c - coordinates (3)
        d - embed dim
        """

        # B      NV           3
        batch, num_vertices, num_coors, device = *vertices.shape, vertices.device
        # B   nf          nvg = 3
        _, num_faces, num_vertices_per_face = faces.shape

        assert self.num_vertices_per_face == num_vertices_per_face

        #rearrange
        # 把face_mask里面标记为true的面的数据进行保留，其余地方的面的数据都被归0
        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        # continuous face coords

        # face_without_pad[b,nf,nfv]可以知道每个面的顶点序号，通过这个序号，去vertices中查表，
        # face_coords = [b,nf,nfv,3]多出来的一维的三个轴存放了序号对应的顶点的坐标
        face_coords = get_at('b [nv] c, b nf mv -> b nf mv c', vertices, face_without_pad)

        # compute derived features and embed
        # 依据每个面的坐标 计算这个面的特征 主要是计算了三个角度、面积、法向量
        # TODO:替换为直接代入
        # derived_features = get_derived_face_features(face_coords)

        # # 将计算出的角度离散化，并进行嵌入
        # # d_angle = b nf n_angle(3)
        # discrete_angle = self.discretize_angle(derived_features['angles'])
        # # b nf n_angle(3) -> b nf n_angle(3) ed
        # angle_embed = self.angle_embed(discrete_angle)
        #
        # # 将计算初的面积离散化，并进嵌入
        # discrete_area = self.discretize_area(derived_features['area'])
        # # b nf 1 -> b nf 1 ed
        # area_embed = self.area_embed(discrete_area)
        #
        # # 将计算出的法向量离散化，并进行嵌入
        # discrete_normal = self.discretize_normals(derived_features['normals'])
        # # b nf 1 -> b nf 1 ed
        # normal_embed = self.normal_embed(discrete_normal)


        # discretize vertices for face coordinate embedding
        # 将坐标离散化
        discrete_face_coords = self.discretize_face_coords(face_coords)
        # b nf nv c -> b nf c
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 or 12 coordinates per face
        # 得到离散化坐标的嵌入向量 b nf c -> b nf c ed(embeding dimension)
        face_coor_embed = self.coor_embed(discrete_face_coords)
        # 将坐标嵌入向量展平
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)')

        # combine all features and project into model dimension

        #face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *')
        face_embed = None
        # 送入线性层，从init_embedding_channel 升到 codebook_dim_channel
        face_embed = self.project_in(face_embed)

        # handle variable lengths by using masked_select and masked_scatter

        # first handle edges
        # needs to be offset by number of faces for each batch

        # 计算batch中每个三角网有效面的个数
        face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum')
        # cumsum计算了Batch的累加有效面数，同时使用pad将累加有效面数整体右移一位
        # 得到当前三角网的偏移量
        face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0)
        face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1')

        # 将face_edges 作不同程度的偏移
        face_edges = face_edges + face_index_offsets
        # 并且只选择有效的边
        face_edges = face_edges[face_edges_mask]

        face_edges = rearrange(face_edges, 'be ij -> ij be')

        # next prepare the face_mask for using masked_select and masked_scatter
        # 得到[b nf]
        orig_face_embed_shape = face_embed.shape[:2]
        # 只选择有效的面,精简化了
        face_embed = face_embed[face_mask]

        # initial sage conv followed by activation and norm
        # 第一层的sage_conv需要act_and_norm
        face_embed = self.init_sage_conv(face_embed, face_edges)

        face_embed = self.init_encoder_act_and_norm(face_embed)

        for conv in self.encoders:
            face_embed = conv(face_embed, face_edges)

        # [b nf channel]
        shape = (*orig_face_embed_shape, face_embed.shape[-1])

        # face_embed.new_zeros(shape) 返回face_embed相同数据类型和设备类型的tensor的纯0tensor，其形状为shape
        # [0...].masked_scatter(mask,source) mask = face_mask,source = face_embed
        # face_mask (shape[b,nf]) --》 face_mask (shape[b,nf,1])
        # 由于face_embed精简化了，所以需要重新填充0
        face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed)

        # 接上注意力网络
        for linear_attn, attn, ff in self.encoder_attn_blocks:
            if exists(linear_attn):
                face_embed = linear_attn(face_embed, mask = face_mask) + face_embed

            face_embed = attn(face_embed, mask = face_mask) + face_embed
            face_embed = ff(face_embed) + face_embed

        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords

    @beartype
    def forward(
            self,
            *,
            vertices: TensorType['b', 'nv', 3, float],
            faces: TensorType['b', 'nf', 'nvf', int],
            faces_feature: TensorType['b', 'nf', 7, float],
            face_edges: TensorType['b', 'e', 2, int] | None = None,
    ):
        if not face_edges:
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, face_coordinates = self.encode(
            vertices = vertices,
            faces = faces,
            face_edges = face_edges,
            face_edges_mask = face_edges_mask,
            face_mask = face_mask,
            return_face_coordinates = True
        )