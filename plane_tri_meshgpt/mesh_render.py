import numpy as np
import math
import os

def combind_mesh_with_rows(path, meshes):
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    translation_distance = 0.5
    obj_file_content = ""

    for row, mesh in enumerate(meshes):
        for r, faces_coordinates in enumerate(mesh):
            numpy_data = faces_coordinates[0].cpu().numpy().reshape(-1, 3)
            numpy_data[:, 0] += translation_distance * (r / 0.2 - 1)
            numpy_data[:, 2] += translation_distance * (row / 0.2 - 1)

            for vertex in numpy_data:
                all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            for i in range(1, len(numpy_data), 3):
                all_faces.append(f"f {i + vertex_offset} {i + 1 + vertex_offset} {i + 2 + vertex_offset}\n")

            vertex_offset += len(numpy_data)

        obj_file_content = "".join(all_vertices) + "".join(all_faces)

    with open(path , "w") as file:
        file.write(obj_file_content)

def save_mesh(path, mesh):
    all_vertices = []
    all_faces = []
    # mesh
    for vertex in mesh:
        all_vertices.append(f"v {vertex[0]} {vertex[1]} 0.0\n")
    for i in range(1, len(mesh), 3):
        all_faces.append(f"f {i} {i + 1} {i + 2}\n")

    obj_file_content = "".join(all_vertices) + "".join(all_faces)
    with open(path, "w") as file:
        file.write(obj_file_content)


def save_mesh_pair(root_dir,mesh_pair):
    ori_data = mesh_pair[0].cpu().numpy().reshape(-1,2)
    pre_data = mesh_pair[1].cpu().numpy().reshape(-1,2)

    ori_obj_filename = "ori_mesh.ply"
    pre_obj_filename = "pre_mesh.ply"
    os.path.join(root_dir,ori_obj_filename)
    ori_path = os.path.join(root_dir,ori_obj_filename)
    pre_path = os.path.join(root_dir,pre_obj_filename)
    save_mesh(ori_path,ori_data)
    save_mesh(pre_path,pre_data)
