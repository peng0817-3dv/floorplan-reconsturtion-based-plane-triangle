from functools import partial
from math import ceil,pi,sqrt
import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Tuple, Callable, List, Dict, Any, Optional, Union
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from torch_geometric.nn.conv import SAGEConv

from einops import reduce, rearrange, pack, unpack,repeat
from einops.layers.torch import Rearrange
from einx   import get_at

from plane_tri_meshgpt.data import derive_face_edges_from_faces

from taylor_series_linear_attention import TaylorSeriesLinearAttn
from local_attention import LocalMHA
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

from torch.cuda.amp import autocast

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)
def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

@beartype
def undiscretize(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo


def derive_angle(x, y, eps = 1e-5):
    z = einsum('... d, ... d -> ...', l2norm(x), l2norm(y))
    # 使用反余弦求角度，则输入应该是余弦值
    return z.clip(-1 + eps, 1 - eps).arccos()

@torch.no_grad()
def get_derived_face_features_from_2d(
    face_coords: TensorType['b', 'nf', 'nvf', 2, float]  # 3 or 4 vertices with 3 coordinates
):
    pad_3d_coords = F.pad(face_coords, (0, 1), mode='constant', value=0.0)
    return get_derived_face_features(pad_3d_coords)

@torch.no_grad()
def get_derived_face_features(
    face_coords: TensorType['b', 'nf', 'nvf', 3, float]  # 3 or 4 vertices with 3 coordinates
):

    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2)

    angles  = derive_angle(face_coords, shifted_face_coords)

    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2)

    cross_product = torch.cross(edge1, edge2, dim = -1)

    normals = l2norm(cross_product)
    area = cross_product.norm(dim = -1, keepdim = True) * 0.5

    return dict(
        angles = angles,
        area = area,
        normals = normals
    )


def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

@beartype
def scatter_mean(
    tgt: Tensor,
    indices: Tensor,
    src = Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-5
):
    """
    todo: update to pytorch 2.1 and try https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
    """
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min = eps)

def divisible_by(num, den):
    return (num % den) == 0
def is_odd(n):
    return not divisible_by(n, 2)

@beartype
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.
) -> Tensor:

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype = dtype, device = device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding = half_width, groups = channels)
    return rearrange(out, 'b c n -> b n c')


class PixelNorm(Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return F.normalize(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = PixelNorm(dim = 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x

class SqueezeExcite(Module):
    def __init__(
        self,
        dim,
        reduction_factor = 4,
        min_dim = 16
    ):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
            nn.Linear(dim_inner, dim),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1')
        )

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min = 1e-5)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')

        return x * self.net(avg)

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out, dropout = dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        res = self.residual_conv(x)
        h = self.block1(x, mask = mask)
        h = self.block2(h, mask = mask)
        h = self.excite(h, mask = mask)
        return h + res


class MeshAutoencoder(Module, PyTorchModelHubMixin):
    @beartype
    def __init__(
            self,
            num_discrete_coors = 128,
            coor_continuous_range:Tuple[float,float] = (-1.,1.),
            dim_coor_embed = 64,
            num_discrete_confidence = 128,
            dim_confidence_embed = 16,
            num_discrete_area=128,
            dim_area_embed=16,
            num_discrete_normals=128,
            dim_normal_embed=64,
            num_discrete_angle=128,
            dim_angle_embed=16,
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
            attn_encoder_depth=0,
            attn_decoder_depth=0,
            local_attn_kwargs: dict = dict(
                dim_head=32,
                heads=8
            ),
            local_attn_window_size=64,
            linear_attn_kwargs: dict = dict(
                dim_head=8,
                heads=16
            ),
            attn_dropout=0.,
            ff_dropout=0.,
            use_linear_attn=True,
            use_residual_lfq=True,  # whether to use the latest lookup-free quantization
            rq_kwargs: dict = dict(
                quantize_dropout=True,
                quantize_dropout_cutoff_index=1,
                quantize_dropout_multiple_of=1,
            ),
            rvq_kwargs: dict = dict(
                kmeans_init=True,
                threshold_ema_dead_code=2,
            ),
            rlfq_kwargs: dict = dict(
                frac_per_sample_entropy=1.,
                soft_clamp_input_value=10.
            ),
            num_quantizers=2,  # or 'D' in the paper
            codebook_size=16384,  # they use 16k, shared codebook between layers
            rvq_stochastic_sample_codes=True,
            checkpoint_quantizer=False,
            decoder_dims_through_depth: Tuple[int, ...] = (
                    128, 128, 128, 128,
                    192, 192, 192, 192,
                    256, 256, 256, 256, 256, 256,
                    384, 384, 384
            ),
            init_decoder_conv_kernel=7,
            resnet_dropout = 0,
            bin_smooth_blur_sigma=0.4,  # they blur the one hot discretized coordinate positions

    ):
        super().__init__()

        self.num_vertices_per_face = 3
        total_coordinates_per_face = self.num_vertices_per_face * 2

        # main face coordinate embedding
        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize,
                                              num_discrete = num_discrete_coors,
                                              continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        # feature embedding
        self.discretize_confidence = partial(discretize,
                                             num_discrete = num_discrete_confidence,
                                             continuous_range = (0., 1.))
        self.confidence_embed = nn.Embedding(num_discrete_confidence, dim_confidence_embed)

        # derived feature embedding

        self.discretize_angle = partial(discretize, num_discrete=num_discrete_angle, continuous_range=(0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete=num_discrete_area,
                                       continuous_range=(0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        # self.discretize_normals = partial(discretize, num_discrete=num_discrete_normals,
        #                                   continuous_range=coor_continuous_range)
        # self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        # 送入网络的向量的特征长度（initial dimension)
        # 6个顶点坐标 * 坐标嵌入维度 + 7个置信度 * 置信度嵌入维度
        init_dim = dim_coor_embed * (2 * self.num_vertices_per_face) +  dim_confidence_embed * 7
        # 6个顶点坐标维度 * 坐标嵌入维度 + 3 * 角度嵌入维度 + 面积嵌入维度
        init_dim = (dim_coor_embed * (2 * self.num_vertices_per_face)
                    + dim_angle_embed * self.num_vertices_per_face
                    + dim_area_embed)

        # project into model dimension
        self.project_in = nn.Linear(init_dim, dim_codebook)

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

        # initial attn block
        self.encoder_attn_blocks = ModuleList([])
        attn_kwargs = dict(
            causal = False,
            prenorm = True,
            dropout = attn_dropout,
            window_size = local_attn_window_size,
        )

        for _ in range(attn_encoder_depth):
            self.encoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(curr_dim, prenorm = True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim = curr_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(curr_dim), FeedForward(curr_dim, glu = True, dropout = ff_dropout))
            ]))

        # --quantize相关--

        # 全连接层，将model的dim（576）映射到 3 * dim_codebook
        self.project_dim_codebook = nn.Linear(curr_dim, dim_codebook * self.num_vertices_per_face)
        # 量化层
        if use_residual_lfq:
            self.quantizer = ResidualLFQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                commitment_loss_weight = 1.,
                **rlfq_kwargs,
                **rq_kwargs
            )
        else:
            self.quantizer = ResidualVQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                shared_codebook = True,
                commitment_weight = 1.,
                stochastic_sample_codes = rvq_stochastic_sample_codes,
                **rvq_kwargs,
                **rq_kwargs
            )
        self.checkpoint_quantizer = checkpoint_quantizer  # whether to memory checkpoint the quantizer

        # --decoder相关

        decoder_input_dim = dim_codebook * 3

        self.decoder_attn_blocks = ModuleList([])

        for _ in range(attn_decoder_depth):
            self.decoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(decoder_input_dim, prenorm = True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim = decoder_input_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(decoder_input_dim), FeedForward(decoder_input_dim, glu = True, dropout = ff_dropout))
            ]))

        init_decoder_dim, *decoder_dims_through_depth = decoder_dims_through_depth
        curr_dim = init_decoder_dim

        assert is_odd(init_decoder_conv_kernel)

        self.init_decoder_conv = nn.Sequential(
            nn.Conv1d(dim_codebook * self.num_vertices_per_face, init_decoder_dim, kernel_size=init_decoder_conv_kernel,
                      padding=init_decoder_conv_kernel // 2),
            nn.SiLU(),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(init_decoder_dim),
            Rearrange('b n c -> b c n')
        )

        self.decoders = ModuleList([])

        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout=resnet_dropout)

            self.decoders.append(resnet_block)
            curr_dim = dim_layer

        self.num_quantizers = num_quantizers
        self.to_coor_logits = nn.Sequential(
            nn.Linear(curr_dim, num_discrete_coors * total_coordinates_per_face),
            Rearrange('... (v c) -> ... v c', v = total_coordinates_per_face)
        )


        # loss related
        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma


    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 'nvf', int],
        faces_feature:    TensorType['b', 'nf', 7, float],
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

        # 计算衍生特征
        derived_features = get_derived_face_features_from_2d(face_coords)

        # 将计算出的置信度特征离散化，并进行嵌入
        discrete_confidence = self.discretize_confidence(faces_feature)
        confidence_embed = self.confidence_embed(discrete_confidence)

        # 计算衍生特征离散化，并进行嵌入
        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)

        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)

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

        face_embed, _ = pack([face_coor_embed, angle_embed,area_embed], 'b nf *')
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
        # return face_embed, discrete_face_coords
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
    def quantize(
            self,
            *,
            faces: TensorType['b', 'nf', 'nvf', int],
            face_mask: TensorType['b', 'n', bool],
            face_embed: TensorType['b', 'nf', 'd', float],
            pad_id=None,
            rvq_sample_codebook_temp=1.
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, num_faces, device = *faces.shape[:2], faces.device
        # 顶点的最大序号
        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)
        # 将维度转换到dim_codebook * 3(顶点个数)的维度
        face_embed = self.project_dim_codebook(face_embed)
        # 拆出成b nf nvf d => b nf 3(顶点个数) dim_codebook
        face_embed = rearrange(face_embed, 'b nf (nvf d) -> b nf nvf d', nvf=self.num_vertices_per_face)
        # 顶点维度 = dim_codebook
        vertex_dim = face_embed.shape[-1]
        # 生成 b * nv * codebook
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device=device)

        # create pad vertex, due to variable lengthed faces

        pad_vertex_id = num_vertices
        vertices = pad_at_dim(vertices, (0, 1), dim=-2, value=0.)

        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # prepare for scatter mean

        faces_with_dim = repeat(faces, 'b nf nvf -> b (nf nvf) d', d=vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim=-2)

        # mask out null vertex token

        mask = torch.ones((batch, num_vertices + 1), device=device, dtype=torch.bool)
        mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict(mask=mask)

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp=rvq_sample_codebook_temp)

        # a quantize function that makes it memory checkpointable

        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        # maybe checkpoint the quantize fn

        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant=False)

        # residual VQ

        quantized, codes, commit_loss = quantize_wrapper_fn((averaged_vertices, quantize_kwargs))

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = get_at('b [n] d, b nf nvf -> b nf (nvf d)', quantized, faces)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        codes_output = get_at('b [n] q, b nf nvf -> b (nf nvf) q', codes, faces)

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf=self.num_vertices_per_face)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss


    @beartype
    def decode(
        self,
        quantized: TensorType['b', 'n', 'd', float],
        face_mask:  TensorType['b', 'n', bool]
    ):
        conv_face_mask = rearrange(face_mask, 'b n -> b 1 n')

        x = quantized

        for linear_attn, attn, ff in self.decoder_attn_blocks:
            if exists(linear_attn):
                x = linear_attn(x, mask = face_mask) + x

            x = attn(x, mask = face_mask) + x
            x = ff(x) + x

        x = rearrange(x, 'b n d -> b d n')
        x = x.masked_fill(~conv_face_mask, 0.)
        x = self.init_decoder_conv(x)

        for resnet_block in self.decoders:
            x = resnet_block(x, mask = conv_face_mask)

        return rearrange(x, 'b d n -> b n d')

    @beartype
    @torch.no_grad()
    def decode_from_codes_to_faces(
        self,
        codes: Tensor,
        face_mask: TensorType['b', 'n', bool] | None = None,
        return_discrete_codes = False
    ):
        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(face_mask):
            face_mask = reduce(codes != self.pad_id, 'b (nf nvf q) -> b nf', 'all', nvf = self.num_vertices_per_face, q = self.num_quantizers)

        # handle different code shapes

        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        # decode

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, 'b (nf nvf) d -> b nf (nvf d)', nvf = self.num_vertices_per_face)

        decoded = self.decode(
            quantized,
            face_mask = face_mask
        )

        decoded = decoded.masked_fill(~face_mask[..., None], 0.)
        pred_face_coords = self.to_coor_logits(decoded)

        pred_face_coords = pred_face_coords.argmax(dim = -1)

        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = self.num_vertices_per_face)

        # back to continuous space

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        # mask out with nan

        continuous_coors = continuous_coors.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        if not return_discrete_codes:
            return continuous_coors, face_mask

        return continuous_coors, pred_face_coords, face_mask

    @beartype
    def forward(
            self,
            *,
            vertices: TensorType['b', 'nv', 3, float],
            faces: TensorType['b', 'nf', 'nvf', int],
            faces_feature: TensorType['b', 'nf', 7, float],
            face_edges: TensorType['b', 'e', 2, int] | None = None,
            rvq_sample_codebook_temp=1.,
            return_loss_breakdown=False,
            return_recon_faces=False,
            only_return_recon_faces=False,
    ):
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)
        # 最大面数、最大面拓扑数
        num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        # 找出哪些面是由于组装batch的时候填充的
        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        # 找出哪些边拓扑是由于组装batch的时候填充的
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        # 特征提取结束
        encoded, face_coordinates = self.encode(
            vertices = vertices,
            faces = faces,
            faces_feature = faces_feature,
            face_edges = face_edges,
            face_edges_mask = face_edges_mask,
            face_mask = face_mask,
            return_face_coordinates = True
        )
        # return encoded, face_coordinates

        quantized, codes, commit_loss = self.quantize(
            face_embed = encoded,
            faces = faces,
            face_mask = face_mask,
            rvq_sample_codebook_temp = rvq_sample_codebook_temp
        )

        # 解码特征
        decode = self.decode(
            quantized,
            face_mask = face_mask
        )
        # 解码特征恢复的预测面
        pred_face_coords = self.to_coor_logits(decode)

        if return_recon_faces or only_return_recon_faces:

            recon_faces = undiscretize(
                pred_face_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )

            recon_faces = rearrange(recon_faces, 'b nf (nvf c) -> b nf nvf c', nvf = self.num_vertices_per_face)
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
            face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')

        if only_return_recon_faces:
            return recon_faces

        # 准备计算重建损失
        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)')

        # reconstruction loss on discretized coordinates on each face
        # they also smooth (blur) the one hot positions, localized label smoothing basically

        with autocast(enabled=False):
            pred_log_prob = pred_face_coords.log_softmax(dim=1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma=self.bin_smooth_blur_sigma)

            # cross entropy with localized smoothing

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim=1)

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r=self.num_vertices_per_face * 2)
            recon_loss = recon_losses[face_mask].mean()

        # calculate total loss

        total_loss = recon_loss + \
                     commit_loss.sum() * self.commit_loss_weight

        # calculate loss breakdown if needed

        loss_breakdown = (recon_loss, commit_loss)
        # some return logic

        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss

            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown