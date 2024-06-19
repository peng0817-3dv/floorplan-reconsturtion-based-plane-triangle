import torch
import os
import shapefile
import numpy as np
from collections import OrderedDict

from plane_tri_meshgpt.data import derive_face_edges_from_faces

# 约定-平面shp文件夹下各子文件的文件名
DATA_VERTICE_FILENAME = "vertexes.shp"
DATA_EDGE_FILENAME = "edges.shp"
DATA_FACE_FILENAME = "poly.shp"

# 约定-vertexes.shp文件中的属性字段名
PROPERTY_VERTICE_CONFIDENCE = "confidence"

# 约定-edges.shp文件中的属性字段名
PROPERTY_EDGE_CONFIDENCE = "confidence"
PROPERTY_EDGE_P1 = "pnt0" # 边所连接的端点1的序号
PROPERTY_EDGE_P2 = "pnt1" # 边所连接的端点2的序号


# 约定-edges.shp文件中的属性字段名
PROPERTY_FACE_CONFIDENCE = "confidence"
PROPERTY_FACE_P1 = "pnt0" # 三角面的顶点1的序号
PROPERTY_FACE_P2 = "pnt1" # 三角面的顶点2的序号
PROPERTY_FACE_P3 = "pnt2" # 三角面的顶点3的序号
PROPERTY_FACE_LABEL = "label" # 三角面的顶点3的序号


def get_vertices_data(vertices_file):
    """
    从点shp文件中提取单个平面图中的所有点信息
    :param vertices_file: 平面图 点shp文件的路径
    :return: (points, confidences) 返回两个list，第一个list装载所有点的坐标对（p_x,p_y）,第二个list装载所有点的置信度
    """
    sf = shapefile.Reader(vertices_file)
    shapes = sf.shapes()
    records = sf.records()
    # shapes中的每一个shape，其points属性中只有一个点，故通过points[0]可以拿到该点
    # 因此points[0][0]拿到该点的x坐标，points[0][1]拿到该点的y坐标，我们用一个元组(px,py)来记录单个点
    vertices = [(float(shape.points[0][0]),float(shape.points[0][1])) for shape in shapes]

    vertices_confidence = [float(record[PROPERTY_VERTICE_CONFIDENCE]) for record in records]
    return vertices, vertices_confidence


def get_edges_data(edge_file):
    """
    从边shp文件中提取单个平面图中的所有边信息
    :param edge_file: 平面图 边shp文件的路径
    :return: (edges, confidences, point_id_to_edge_id),返回两个list和一个dict,第一个list装载所有边的两端点序号，第二个list装载置信度，dict装载点序号到边的快速索引
    """
    sf = shapefile.Reader(edge_file)
    shapes = sf.shapes()
    records = sf.records()
    edges = [(record[PROPERTY_EDGE_P1],record[PROPERTY_EDGE_P2]) for record in records]

    # point_id_to_edges_id = { point_id : [edge1_id,edge2_id,...] }
    # 如此，可以根据点的id快速锁定该点连接的边的id，从而，我们可以依据两个点id快速锁定一个边的id
    point_id_to_edges_id = {}
    for index,record in enumerate(records):
        point1_id = record[PROPERTY_EDGE_P1]
        point2_id = record[PROPERTY_EDGE_P2]

        # 以防止首次访问point1_id，没有找到key出错
        if point1_id in point_id_to_edges_id:
            point_id_to_edges_id[point1_id].append(index)
        else:
            point_id_to_edges_id[point1_id] = [index]
        # 同理
        if point2_id in point_id_to_edges_id:
            point_id_to_edges_id[point2_id].append(index)
        else:
            point_id_to_edges_id[point2_id] = [index]

    edges_confidence = [float(record[PROPERTY_EDGE_CONFIDENCE]) for record in records]
    # 知道可以用一个循环解决三个lit的生成，但是上面这么写简洁一些，可读性好一点
    return edges, edges_confidence, point_id_to_edges_id


def get_faces_data(face_file):
    """
    从面shp文件中提取单个平面中的所有三角形的信息
    :param face_file: 平面图 面shp文件的路径
    :return: (faces,faces_confidence, faces_label),返回三个list，第一个list装载所有三角形的顶点序号，第二个list装载所有三角形的置信度，第三个是该三角形的label
    """
    sf = shapefile.Reader(face_file)
    shapes = sf.shapes()
    records = sf.records()
    faces = [(record[PROPERTY_FACE_P1], record[PROPERTY_FACE_P2], record[PROPERTY_FACE_P3]) for record in records]
    faces_confidences = [float(record[PROPERTY_FACE_CONFIDENCE]) for record in records]
    faces_label = [record[PROPERTY_FACE_LABEL] for record in records]
    return faces, faces_confidences, faces_label


def get_edge_id_of_face(face: tuple, point_id_to_edge_id):
    """
    找出该面对应的edge编号
    :param face:
    :param point_id_to_edge_id:
    :return:
    """
    v1,v2,v3 = face[0],face[1],face[2]
    v1_relate_edges = point_id_to_edge_id[v1]
    v2_relate_edges = point_id_to_edge_id[v2]
    v3_relate_edges = point_id_to_edge_id[v3]

    # 根据两个顶点之间关联的边的交集锁定 两个点之间的边
    e1 = list(set(v1_relate_edges).intersection(v2_relate_edges))
    e2 = list(set(v2_relate_edges).intersection(v3_relate_edges))
    e3 = list(set(v3_relate_edges).intersection(v1_relate_edges))
    # 两个点之间的边 应该有且仅有一条
    assert (len(e1) == 1)
    assert (len(e2) == 1)
    assert (len(e3) == 1)
    e1 = e1[0]
    e2 = e2[0]
    e3 = e3[0]
    return e1,e2,e3


def get_plane_mesh(obj_root_path):
    """
    将一个平面图从shp文件格式 转化为若干个np数据结构
    :param obj_root_path: 一个平面图场景文件夹
    :return:[sorted_vertices, sorted_faces, face_feature]
    """
    vertices_file = os.path.join(obj_root_path,DATA_VERTICE_FILENAME)
    vertices, v_confidence = get_vertices_data(vertices_file)
    edge_file = os.path.join(obj_root_path,DATA_EDGE_FILENAME)
    edges, e_confidence, point_id_to_edge_id = get_edges_data(edge_file)
    face_file = os.path.join(obj_root_path,DATA_FACE_FILENAME)
    faces, f_confidence, labels = get_faces_data(face_file)

    # face_feature = [face_id:[f_confidence]]
    face_feature = [0] * len(faces) # 构造faces同等长度的列表，初值填充0
    for index,(v1,v2,v3) in enumerate(faces):
        # 找到每个面的顶点所关联的边
        e1,e2,e3 = get_edge_id_of_face(faces[index],point_id_to_edge_id)
        face_feature[index] = [v_confidence[v1],v_confidence[v2],v_confidence[v3],
                               e_confidence[e1],e_confidence[e2],e_confidence[e3],
                               f_confidence[index]]

    # 参考 marcus_meshgpt 对点的处理
    # 归一化
    centered_vertices = vertices - np.mean(vertices, axis=0)
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)  # Limit vertices to [-0.95, 0.95]

    min_y = np.min(vertices[:, 1])
    difference = -0.95 - min_y
    vertices[:, 1] += difference

    # 去重
    seen = OrderedDict()
    for point in vertices:
        key = tuple(point)
        if key not in seen:
            seen[key] = point
    # 得到所有非重复点
    unique_vertices = list(seen.values())


    def sort_vertices(vertex):
        return vertex[1], vertex[0]
    # 排序以y轴优先？
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)
    # seen_sorted_vertices = np.array(sorted_vertices)
    # 原来点的序号
    vertices_as_tuples = [tuple(v) for v in vertices]
    # 排序去重后的点的序号
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]
    # 建立hash映射
    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples) for
                  new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples) if
                  vertex_tuple == sorted_vertex_tuple}
    # 依次hash映射将面的索引进行相应更新
    # 面的索引进行相应更新
    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces]

    sorted_faces = [sorted(sub_arr) for sub_arr in reindexed_faces]
    # 点的重新排序、和面索引的更新，不会影响面特征
    return np.array(sorted_vertices), np.array(sorted_faces), np.array(face_feature)



def load_filename(directory, variations = -1):
    """
    载入shp文件，输出numpy变量
    :param directory: shp文件所在根目录
    :param variations: 载入数量，当数量为-1时，载入全部
    :return:
    """
    obj_datas = []

    for count, filename in  enumerate(os.listdir(directory)):
        # 如果指定了固定数量，则载入一定程度的文件讲会提前结束
        if(variations >= 0):
            if count >= variations:
                break
        obj_root_path = os.path.join(directory, filename)
        # 确保访问的只是文件夹
        if not os.path.isdir(obj_root_path):
            continue

        # 得到点集，面集，面特征集
        vertices, faces, faces_feature = get_plane_mesh(obj_root_path)
        # 面可以先送入GPU,加速面拓扑关系的计算
        faces = torch.tensor(faces.tolist(),dtype=torch.long).to("cuda")
        faces_feature = torch.tensor(faces_feature.tolist(),dtype=torch.float32).to("cuda")
        # 面的拓扑关系，和面的三条边不是一个概念，注意别混淆
        face_edges = derive_face_edges_from_faces(faces)

        obj_data = {
            "vertices":torch.tensor(vertices.tolist(),dtype=torch.float).to("cuda"),
            "faces":faces, # 已送入gpu
            "faces_feature":faces_feature, # 已送入gpu
            "face_edges":face_edges,
        }
        obj_datas.append(obj_data)

    print(f"[create_mesh_dataset] Returning {len(obj_datas)} meshes")
    return obj_datas


