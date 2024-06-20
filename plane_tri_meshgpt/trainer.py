import torch
from torch.nn import Module
from torch.nn.modules.module import T
from torch.utils.data import Dataset, DataLoader

from pytorch_custom_utils import (
    get_adam_optimizer,
    OptimizerWithWarmupSchedule
)

from contextlib import nullcontext, contextmanager

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from torch.optim.lr_scheduler import _LRScheduler


from functools import partial

from beartype import beartype
from tqdm import tqdm

from plane_tri_meshgpt.meshgpt_pytorch import MeshAutoencoder
from plane_tri_meshgpt.data import custom_collate
from beartype.typing import Optional, Tuple, Type, List




DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)



class MeshAutoencoderTrainer(Module):
    @beartype
    def __init__(
        self,
        model: MeshAutoencoder,
        dataset: Dataset,
        batch_size: int,
        grad_accum_every: int,
        data_kwargs: Tuple[str, ...] = ('vertices', 'faces', 'faces_channel', 'face_edges'),  # data_kwargs默认就是一个元组，该元组存放3个字符串
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        optimizer_kwargs: dict = dict(),
        scheduler: Type[_LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps=1000,
        max_grad_norm: float | None = None,
        use_wandb_tracking=False,
        accelerator_kwargs: dict = dict(),
    ):
        super().__init__()

        self.model = model

        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            drop_last = True,
            collate_fn = partial(custom_collate, pad_id = model.pad_id),

        )

        self.grad_accum_every = grad_accum_every

        self.data_kwargs = data_kwargs

        # accelerator

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)


        # optimizer
        self.optimizer = OptimizerWithWarmupSchedule(
            accelerator = self.accelerator,
            optimizer = get_adam_optimizer(model.parameters(), lr = learning_rate, wd = weight_decay, **optimizer_kwargs),
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )


    def train(
        self,
        num_epochs,
    ):
        self.model.train()
        for epoch in range(num_epochs):
            # 一轮训练
            total_epoch_loss, total_epoch_recon_loss, total_epoch_commit_loss = 0.0, 0.0, 0.0
            progress_bar = tqdm(enumerate(self.dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', total=len(self.dataloader))
            for batch_idx, batch in progress_bar:
                # grad_accum_every被设置为4,每4个batch is_last被置为True
                is_last = (batch_idx+1) % self.grad_accum_every == 0
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                # 如果batch是一个元组，则认为该元组是三个子batch的排列 'vertices', 'faces', 'face_edges'
                # 将forward_kwargs变为一个带名称的字典
                if isinstance(batch, tuple):
                    forward_kwargs = dict(zip(self.data_kwargs, batch))
                    # 如果batch是一个字典，默认应该是这个，因为dataset从load_filename函数中得到的就是一个字典
                elif isinstance(batch, dict):
                    forward_kwargs = batch
                # 去掉'texts' key 所在的元素
                # maybe_del(forward_kwargs, 'texts', 'text_embeds')

                with self.accelerator.autocast(), maybe_no_sync():
                    total_loss, (recon_loss, commit_loss) = self.model(
                        **forward_kwargs,
                        return_loss_breakdown = True
                    )
                    self.accelerator.backward(total_loss / self.grad_accum_every)

                current_loss = total_loss.item()
                total_epoch_loss += current_loss
                total_epoch_recon_loss += recon_loss.item()
                total_epoch_commit_loss += commit_loss.sum().item()

                progress_bar.set_postfix(loss=current_loss, recon_loss=round(recon_loss.item(), 3),
                                         commit_loss=round(commit_loss.sum().item(), 4))

                # is_last每四个batch被设置为True，意思是每4个batch梯度下降一次？
                if is_last or (batch_idx + 1 == len(self.dataloader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

