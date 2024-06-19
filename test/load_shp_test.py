import os.path
import random
import unittest
import pytest
from utils import load_shp
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot,axes
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

class LoadShpTest(unittest.TestCase):

    def setUp(self):
        self.dataset_root = 'G:\workspace_plane2DDL\label_shp_root'
        self.scene_root = os.path.join(self.dataset_root,'scene_00177')
        self.vertice_file = os.path.join(self.scene_root,load_shp.DATA_VERTICE_FILENAME)
        self.edge_file = os.path.join(self.scene_root,load_shp.DATA_EDGE_FILENAME)
        self.face_file = os.path.join(self.scene_root,load_shp.DATA_FACE_FILENAME)

        self.points,self.v_conf = load_shp.get_vertices_data(self.vertice_file)
        self.edges,self.e_conf,self.point_id_to_edge_id = load_shp.get_edges_data(self.edge_file)
        self.faces,self.f_conf,self.label = load_shp.get_faces_data(self.face_file)
        # self.test_result =
    def test_get_vertices_data(self):

        points_x = np.array([p[0] for p in self.points])
        points_y = np.array([p[1] for p in self.points])
        fig,ax = plt.subplots()
        ax.scatter(points_x,points_y,c = self.v_conf,cmap = 'RdYlGn')
        # plt.show()
        plt.savefig('./load_shp_test/vertice.png')
        self.assertTrue(True)

    def test_get_edges_data(self):

        lines = [[self.points[edge[0]],self.points[edge[1]]] for edge in self.edges]
        lines = np.array(lines)
        e_conf = np.array(self.e_conf)
        lines = LineCollection(lines,array = e_conf,cmap= 'RdYlGn')

        fig,ax = plt.subplots()
        ax.add_collection(lines)
        ax.autoscale()
        plt.savefig('./load_shp_test/edge.png')
        self.assertTrue(True)

    def test_get_faces_data(self):
        points = self.points

        tris = np.array([[points[face[0]],points[face[1]],points[face[2]]] for face in self.faces])
        f_conf = np.array(self.f_conf)

        tris = PolyCollection(tris, array = f_conf, cmap='RdYlGn')
        fig,ax = plt.subplots()
        ax.add_collection(tris)
        ax.autoscale()
        plt.savefig('./load_shp_test/face.png')
        self.assertTrue(True)


    def test_get_edge_id_of_face(self):
       # random_id = random.randint(0,len(self.faces))
        random_face = self.faces[random.randint(0,len(self.faces))]
        e1,e2,e3 = load_shp.get_edge_id_of_face(random_face,self.point_id_to_edge_id)
        vertices = set(self.edges[e1]).union(set(self.edges[e2])).union(set(self.edges[e3]))
        self.assertSetEqual(vertices,set(random_face))

    def test_get_plane_mesh(self):
        sorted_vertices, sorted_faces, face_feature = load_shp.get_plane_mesh(obj_root_path=self.scene_root)

        fig,ax = plt.subplots()

        # ax.scatter(sorted_vertices[:,0],sorted_vertices[:,1],cmap = 'RdYlGn')
        # plt.savefig('./load_shp_test/sorted_vertice.png')

        tris = np.array([[sorted_vertices[face[0]],sorted_vertices[face[1]],sorted_vertices[face[2]]] for face in sorted_faces])
        f_conf = np.array(self.f_conf)
        tris = PolyCollection(tris, array = f_conf, cmap='RdYlGn')
        ax.add_collection(tris)
        ax.autoscale()
        plt.savefig('./load_shp_test/sorted_face.png')

        self.assertTrue(True)

    def test_load_filename(self):
        load_num = 3
        obj_datas = load_shp.load_filename(self.dataset_root,variations=load_num)
        self.assertEqual(len(obj_datas),load_num)

        obj_data = obj_datas[0]
        face_edges = obj_data['face_edges']
        vertice = obj_data['vertices'].cpu()
        faces = obj_data['faces']
        centers = []
        for face in faces:
            v1 = vertice[face[0]].cpu()
            v2 = vertice[face[1]].cpu()
            v3 = vertice[face[2]].cpu()
            center = (v1 + v2 + v3) / 3
            centers.append(center)

        tris = np.array([[vertice[face[0]],vertice[face[1]],vertice[face[2]]] for face in faces])

        tris = PolyCollection(tris,facecolors= 'r')


        lines = [[centers[edge[0]], centers[edge[1]]] for edge in face_edges]
        lines = np.array(lines)
        lines = LineCollection(lines)
        fig,ax = plt.subplots()
        ax.add_collection(tris)
        ax.add_collection(lines)
        ax.autoscale()
        plt.savefig('./load_shp_test/face_edges.png')
        self.assertTrue(True)



