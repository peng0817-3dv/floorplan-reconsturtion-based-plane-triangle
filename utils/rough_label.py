import os
import os.path
import shutil

import shapefile
from shapely import Polygon,Point


ANNO_GT_BOUND = 'GT_bound'
ANNO_GT_POLY = 'GT_room_poly'


def rough_label(
    tri_save_path,
    anno_save_path,
    need_copy_anno = True,
):
    scenes = os.listdir(tri_save_path)
    for scene in scenes:
        tri_path = os.path.join(tri_save_path, scene)
        rough_label_path = os.path.join(tri_path, "label_poly.shp")
        if os.path.exists(rough_label_path):
            print(f"{scene} rough label already exist,will not rough labeling again")
        else:
            # rough labeling
            anno_path = os.path.join(anno_save_path, scene)
            if os.path.exists(anno_path):
                wrap_core(tri_path, anno_path, need_copy=need_copy_anno)
            else:
                print("don't find anno file,rough labeling is not execute")
        print(f"...{scene} end")


def wrap_core(mesh_path,anno_path, need_copy = True):
    rough_label_core(mesh_path,anno_path)
    if need_copy:
        copy_GT_poly_and_bound_to_labelFile(mesh_path,anno_path)

def rough_label_core(mesh_path, anno_path):
    # 设置路径
    tris_shp_path = os.path.join(mesh_path,'poly')
    rooms_shp_path = os.path.join(anno_path,'GT_room_poly')
    bound_shp_path = os.path.join(anno_path,'GT_bound')
    # 打开shp文件
    tris_shp = shapefile.Reader(tris_shp_path)
    rooms_shp = shapefile.Reader(rooms_shp_path)
    bound_shp = shapefile.Reader(bound_shp_path)

    # 将注解文件从shp文件转化为polygon数据结构
    # 房间多边形集
    rooms_poly_list = []
    rooms_records = rooms_shp.records()
    for room_shp in rooms_shp.shapes():
        rooms_poly_list.append(Polygon(room_shp.points))
    # 边界多边形
    bound_poly = Polygon(bound_shp.shape(0).points)

    # 打开三角形格网的shp文件
    tris = tris_shp.shapes()
    # 暂存待标图形的属性，后面将会修改
    tris_records = tris_shp.records()

    # 用于统计的变量
    out_bound_count = 0
    label_count = 0
    no_label_count = 0

    # 使用多边形重叠面积大于80%来粗暴打标
    for i,tri in enumerate(tris):
        tri_poly = Polygon(tri.points)
        # 先检查，当前三角形是否落在边界内
        if (tri_poly.intersection(bound_poly).area/ tri_poly.area < 0.2):
            out_bound_count += 1
            tris_records[i]['label'] = -2
            continue

        is_find_label = False
        for j,room_poly in enumerate(rooms_poly_list):
            if tri_poly.intersects(room_poly):
                # 三角形和该房间相交,且相交部分大于80%
                if (tri_poly.intersection(room_poly).area / tri_poly.area > 0.8):
                    is_find_label = True
                    tris_records[i]['label'] = j
                    break

        if is_find_label:
            label_count += 1
            continue
        else:
            # 所有房间遍历完，发现该三角形都很难分辨
            no_label_count += 1
            tris_records[i]['label'] = -1

    # 输出统计数据
    print(f"out of bound:{out_bound_count},label:{label_count},hard to label:{no_label_count}")

    # 写出到新shp中
    label_tris_shp_path = os.path.join(mesh_path, 'label_poly')
    w_tri = shapefile.Writer(label_tris_shp_path)
    w_tri.fields = tris_shp.fields[1:]
    shape_count = 0
    for shaperec in tris_shp.iterShapeRecords():
        w_tri.record(*tris_records[shape_count])
        w_tri.shape(shaperec.shape)
        shape_count += 1
    w_tri.close()
    print('rough_labeling done')

def copy_GT_poly_and_bound_to_labelFile(mesh_path, anno_path):
    gt_bound_dbf = ANNO_GT_BOUND + '.dbf'
    gt_bound_dbf = os.path.join(anno_path,gt_bound_dbf)
    gt_bound_shp = ANNO_GT_BOUND + '.shp'
    gt_bound_shp = os.path.join(anno_path, gt_bound_shp)
    gt_bound_shx = ANNO_GT_BOUND + '.shx'
    gt_bound_shx = os.path.join(anno_path, gt_bound_shx)

    gt_poly_dbf = ANNO_GT_POLY + '.dbf'
    gt_poly_dbf = os.path.join(anno_path, gt_poly_dbf)
    gt_poly_shp = ANNO_GT_POLY + '.shp'
    gt_poly_shp = os.path.join(anno_path, gt_poly_shp)
    gt_poly_shx = ANNO_GT_POLY + '.shx'
    gt_poly_shx = os.path.join(anno_path, gt_poly_shx)

    shutil.copy(gt_bound_dbf,mesh_path)
    shutil.copy(gt_bound_shp,mesh_path)
    shutil.copy(gt_bound_shx,mesh_path)
    shutil.copy(gt_poly_dbf,mesh_path)
    shutil.copy(gt_poly_shp,mesh_path)
    shutil.copy(gt_poly_shx,mesh_path)

if __name__ == '__main__':
    anno_shp_root = "G:/workspace_plane2DDL/anno_shp_root"

    # 说是label_shp，其实叫待label更合适理解，不过由于粗标注是原地标注，程序执行完后该文件夹下确实是label_shp
    label_shp_root = "G:/workspace_plane2DDL/label_shp_root"

    rough_label(
        tri_save_path=label_shp_root,
        anno_save_path=anno_shp_root,
    )