import torch
from torch import is_tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from torchtyping import TensorType
from einops import rearrange,reduce



def derive_face_edges_from_faces(
    faces: TensorType['b', 'nf', 3, int],
    pad_id = -1,
    neighbor_if_share_one_vertex = False,
    include_self = True
) -> TensorType['b', 'e', 2, int]:

    # 如果faces张量的维度只有2，代表传入的是单个faces，而非一个batch的faces
    is_one_face, device = faces.ndim == 2, faces.device

    # 调整单个face张量的形状，使得可以和batch faces一同处理
    if is_one_face:
        faces = rearrange(faces, 'nf c -> 1 nf c') # 代表batch中只有一张图

    # faces的第二维度代表了 一批平面图中 某张最多三角形数的mesh，因为其他的mesh数量不够的都被填充
    max_num_faces = faces.shape[1]

    # 共享一个顶点的平面就视为邻居，还是共享两个？
    face_edges_vertices_threshold = 1 if neighbor_if_share_one_vertex else 2

    # 简而言之，假使n == max_num_faces,就是生成一个n*n*2的张量，记录了n*n网格上的所有坐标
    all_edges = torch.stack(torch.meshgrid(
        torch.arange(max_num_faces, device = device),
        torch.arange(max_num_faces, device = device),
    indexing = 'ij'), dim = -1)
    # faces != pad_id 利用张量的广播机制，返回一个和faces相同形状的bool张量，其中面为填充id的那一层都为False
    # face_mask (b,nf),得到了哪些面是非填充的，哪些面是填充的
    face_masks = reduce(faces != pad_id, 'b nf c -> b nf', 'all')
    # 得到哪些边对是有意义的，哪些边对是无意义的
    face_edges_masks = rearrange(face_masks, 'b i -> b i 1') & rearrange(face_masks, 'b j -> b 1 j')

    face_edges = []

    # 迭代一个批次里每个面
    for face, face_edge_mask in zip(faces, face_edges_masks):
        # 比较难以理解，但从结果反推就是利用张量的广播机制快速找到面之间的相邻点
        shared_vertices = rearrange(face, 'i c -> i 1 c 1') == rearrange(face, 'j c -> 1 j 1 c')
        num_shared_vertices = shared_vertices.any(dim = -1).sum(dim = -1)

        # 计算出每个面之间是否有相邻关系
        is_neighbor_face = (num_shared_vertices >= face_edges_vertices_threshold) & face_edge_mask

        if not include_self:
            is_neighbor_face &= num_shared_vertices != 3
        # 得到相邻关系的坐标组，即拓扑关系
        face_edge = all_edges[is_neighbor_face]
        face_edges.append(face_edge)

    # 由于迭代处理汇总的结果中face_edges中的每个face_edge的长度都是不同的，为了后续方便批处理，将其填充到和最长的等长，
    face_edges = pad_sequence(face_edges, padding_value = pad_id, batch_first = True)

    if is_one_face:
        face_edges = rearrange(face_edges, '1 e ij -> e ij')

    return face_edges


# custom collater

def first(it):
    return it[0]
def custom_collate(data, pad_id = -1):
    """
    供Dataloader对Dataset进行批量化时，对不足数据的自定义填充函数
    https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    :param data:
    :param pad_id:
    :return:
    """
    is_dict = isinstance(first(data), dict)

    if is_dict:
        # 则此时data格式为：[{k1:v1,k2:v2, ...}{k1:v1,k2:v2, ...} ...]
        keys = first(data).keys()
        # 整理data为list [[v1,v2, ...][v1,v2, ...] ...]
        data = [d.values() for d in data]

    output = []
    # *data ： 将data解包 为一个个list [v1,v2, ...], [v1,v2, ...], ...
    # zip : 将可迭代对象作为参数，将每个对象中的对应的元素打包成一个个元组
    # 则 zip(*data) => [(v1,v1,...)(v2,v2,...) ...]
    for datum in zip(*data):
        if is_tensor(first(datum)):
            # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            #如果不是张量，则不提供填充支持
            datum = list(datum)
            output.append(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output

def custom_collate_with_feature(
        data,
        pad_id = -1,
        feature_key = 'faces_feature',
        feature_pad = 0.0,
):
    keys = first(data).keys()
    data = [d.values() for d in data]
    data_collect = zip(*data)
    table = dict(zip(keys, data_collect))

    output = []
    for key, datum in table.items():
        if key == feature_key:
            datum = pad_sequence(datum, batch_first=True, padding_value= feature_pad)
        else:
            datum = pad_sequence(datum, batch_first=True, padding_value=pad_id)
        output.append(datum)

    output = dict(zip(keys, output))
    return output