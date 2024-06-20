import os.path
import random
import unittest
from plane_tri_meshgpt.data import custom_collate,custom_collate_with_feature
from utils.load_shp import load_filename
from plane_tri_meshgpt.planetri_dataset import PlaneTriDataset

TEST_DATASET_RESOURCE = 'G:\workspace_plane2DDL\label_shp_root'
TEST_DATASET_NPZ_BACKUP = '../demo/demo.npz'
# TEST_SCENE_ID = 'scene_00177'
# TEST_RES_SAVE_FILEDIR = './load_shp_test/'
class DataTest(unittest.TestCase):
    def test_custom_collate(self):
        dataset = PlaneTriDataset.load(TEST_DATASET_NPZ_BACKUP)
        batch_index = random.sample(range(len(dataset)), 4)

        # batch = custom_collate([dataset[idx] for idx in batch_index]) # 对照组
        batch = custom_collate_with_feature([dataset[idx] for idx in batch_index])

        faces_feature = batch['faces_feature']
        bool_faces_feature = faces_feature.ge(0.0) & faces_feature.le(100.0)
        self.assertTrue(bool_faces_feature.all().item())


