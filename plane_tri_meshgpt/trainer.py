import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from functools import partial

from beartype import beartype

from plane_tri_meshgpt.meshgpt_pytorch import MeshAutoencoder
from plane_tri_meshgpt.data import custom_collate


class MeshAutoencoderTrainer(Module):
    @beartype
    def __init__(
        self,
        model: MeshAutoencoder,
        dataset: Dataset,
        batch_size: int,

    ):
        super().__init__()

        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            drop_last = True,
            collate_fn = partial(custom_collate, pad_id = model.pad_id),
        )
