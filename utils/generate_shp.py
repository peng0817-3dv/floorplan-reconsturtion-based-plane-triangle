import os


COLLECT_POINT_CLOUD_PATH = "D:/point_cloud"
CPP_PROCESS_EXE_PATH = "F:/DIP/DipTools_Indoor/x64/Release/ExportMeshGptFeature.exe"
LABEL_ROOT_PATH = "D:/triplane"

def generate_floorplan_by_pointcloud(
        pcl_load_path,
        tri_save_path,
        cpp_exe_path,
        cpp_process_override = False
):
    """

    用于将成批的点云调用实验室前人工作【点云提取特征线，生成平面三角格网】，生成成批的shp文件
    如果点云为ply文件，将依赖cloud compare的命令行工具转化为las
    每个点云都会对应的生成一组shp文件，包含vertexes.shp[顶点文件]、edges.shp[边文件]、poly.shp[面文件]
    由于是成批处理，因此，该程序对pcl_load_path下的点云文件架构约定如下：

    pcl_load_path
    |--scene001
    |   |-point_cloud.las
    |--scene002
    |--scene003
    |--...

    我们要求每个点云都以一个文件夹隔离开，文件夹名可以是这个点云的名称，但是具体的点云文件名称要求是point_cloud.las
    我们会相应的在tri_save_path下生成相同名称的文件夹，并将各类属于该点云而生成的shp文件放置其中，如下：

    tri_save_path
    |--scene001
    |   |-vertexes.shp
    |   |-edges.shp
    |   |-poly.shp
    |--scene002
    |--scene003
    |--...

    至于cpp_process_override参数，即检查tri_save_path是否已存在同名文件夹，然后依据该参数决定是否覆写
    当然这个参数和具体的cpp_exe紧耦合

    :param pcl_load_path:点云源路径
    :param tri_save_path:生成平面存放路径
    :param cpp_exe_path:处理cpp程序
    :param cpp_process_override:是否覆写
    :return:None

    """
    scenes = os.listdir(pcl_load_path)
    for scene in scenes:
        print(f"{scene} generate shp.")
        source_las_path = os.path.join(pcl_load_path,scene,'point_cloud.las')
        if not os.path.exists(source_las_path):
            # 如果指定场景文件夹下的点云后缀是ply,则我们需要将其转换为las后缀，方便后续
            mid_ply_path = os.path.join(pcl_load_path,scene,'point_cloud.ply')
            if not  os.path.exists(mid_ply_path):
                continue
            # 必须要将cloudcompare添加到环境路径中去，我们才可以在命令行中调用CloudCompare指令
            cmd = "CloudCompare -SILENT -O {0} -C_EXPORT_FMT LAS -NO_TIMESTAMP -SAVE_CLOUDS".format(mid_ply_path)
            os.system(cmd)
            # 转换成我们需要的las格式后，ply点云可以及时清除，节省空间
            os.remove(mid_ply_path)
        target_tri_path = os.path.join(tri_save_path,scene)
        override_param = 'true' if cpp_process_override else 'false'
        cpp_process_cmd = f"{cpp_exe_path} -i {source_las_path} -o {target_tri_path} -n {override_param}"
        os.system(cpp_process_cmd)

if __name__ == '__main__':
    generate_floorplan_by_pointcloud(
        pcl_load_path=COLLECT_POINT_CLOUD_PATH,
        tri_save_path=LABEL_ROOT_PATH,
        cpp_exe_path=CPP_PROCESS_EXE_PATH,
    )