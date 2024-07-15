import unittest
import os
TEST_DATASET_RESOURCE = 'G:\workspace_plane2DDL\label_shp_root'
TEST_DATASET_NPZ_BACKUP = '../demo/demo.npz'

project_name = "demo_mesh"
DATA_RESOURCE_PATH = 'G:\workspace_plane2DDL\label_shp_root'
working_dir = f'./{project_name}'

class LoadShpTest(unittest.TestCase):
    def test_path_join(self):
        ori_samples = range(30)
        for i, ori in enumerate(ori_samples):
        #    pre = pre_samples[i]
            root_dir = os.path.join(f'{working_dir}', 'mesh', str(i))
            print(root_dir)
