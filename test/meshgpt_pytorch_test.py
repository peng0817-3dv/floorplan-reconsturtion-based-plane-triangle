import unittest
import random
from functools import partial

from einops import rearrange,reduce
from einx   import get_at

from plane_tri_meshgpt.data import custom_collate_with_feature
from plane_tri_meshgpt.meshgpt_pytorch import MeshAutoencoder
from plane_tri_meshgpt.planetri_dataset import PlaneTriDataset
from plane_tri_meshgpt.meshgpt_pytorch import get_derived_face_features_from_2d
from torch.utils.data import Dataset, DataLoader
from torch.nn import  functional as F

TEST_DATASET_NPZ_BACKUP = '../demo/demo.npz'

class MeshgptPytorchTest(unittest.TestCase):
    def setUp(self):
        dataset = PlaneTriDataset.load(TEST_DATASET_NPZ_BACKUP)
        dataloader = DataLoader(
            dataset,
            batch_size= 4,
            collate_fn= partial(custom_collate_with_feature, pad_id = -1),
        )
        # 取一份数据
        data_batch = next(iter(dataloader))
        self.data_batch = data_batch

    def test_forward(self):
        dataset = PlaneTriDataset.load(TEST_DATASET_NPZ_BACKUP)
        dataloader = DataLoader(
            dataset,
            batch_size= 4,
            collate_fn= partial(custom_collate_with_feature, pad_id = -1),
        )
        # 取一份数据
        data_batch = next(iter(dataloader))
        model = MeshAutoencoder().to("cuda")
        total_loss, (recon_loss, commit_loss) = model(**data_batch,
                                         return_loss_breakdown=True
                                         )
        print(total_loss)
        self.assertTrue(True)

    def test_discretize_confidence(self):
        dataset = PlaneTriDataset.load(TEST_DATASET_NPZ_BACKUP)
        dataloader = DataLoader(
            dataset,
            batch_size= 4,
            collate_fn= partial(custom_collate_with_feature, pad_id = -1),
        )
        # 取一份数据
        data_batch = next(iter(dataloader))
        faces_feature = data_batch['faces_feature']

        model = MeshAutoencoder().to("cuda")
        discrete_confidence = model.discretize_confidence(faces_feature)
        confidence_embed = model.confidence_embed(discrete_confidence)
        self.assertTrue(True)

    def test_discretize_face_coords(self):
        dataset = PlaneTriDataset.load(TEST_DATASET_NPZ_BACKUP)
        dataloader = DataLoader(
            dataset,
            batch_size= 4,
            collate_fn= partial(custom_collate_with_feature, pad_id = -1),
        )
        # 取一份数据
        data_batch = next(iter(dataloader))
        vertices = data_batch['vertices']
        faces_feature = data_batch['faces_feature']
        faces = data_batch['faces']
        face_mask = reduce(faces != -1, 'b nf c -> b nf', 'all')
        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)
        face_coords = get_at('b [nv] c, b nf mv -> b nf mv c', vertices, face_without_pad)
        model = MeshAutoencoder().to("cuda")
        discrete_face_coords = model.discretize_face_coords(face_coords)
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') #6 coordinates per face
        face_coor_embed = model.coor_embed(discrete_face_coords)
        self.assertTrue(True)

    def test_get_derived_face_features(self):
        vertices = self.data_batch['vertices']
        faces = self.data_batch['faces']
        face_mask = reduce(faces != -1, 'b nf c -> b nf', 'all')
        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        face_coords = get_at('b [nv] c, b nf mv -> b nf mv c', vertices, face_without_pad)
        features = get_derived_face_features_from_2d(face_coords)
        print(features)
        self.assertTrue(True)
