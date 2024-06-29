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
import matplotlib.pyplot as plt
from tqdm import tqdm

from plane_tri_meshgpt.meshgpt_pytorch import MeshAutoencoder
from plane_tri_meshgpt.data import custom_collate,custom_collate_with_feature
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
            collate_fn = partial(custom_collate_with_feature, pad_id = model.pad_id),
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
        stop_at_loss = None,
        display_loss_graph = False
    ):
        epoch_losses, epoch_recon_losses, epoch_commit_losses = [] , [],[]

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

            avg_recon_loss = total_epoch_recon_loss / len(self.dataloader)
            avg_commit_loss = total_epoch_commit_loss / len(self.dataloader)
            avg_epoch_loss = total_epoch_loss / len(self.dataloader)

            epoch_losses.append(avg_epoch_loss)
            epoch_recon_losses.append(avg_recon_loss)
            epoch_commit_losses.append(avg_commit_loss)

            epochOut = f'Epoch {epoch + 1} average loss: {avg_epoch_loss} recon loss: {avg_recon_loss:.4f}: commit_loss {avg_commit_loss:.4f}'

            if len(epoch_losses) >= 4 and avg_epoch_loss > 0:
                avg_loss_improvement = sum(epoch_losses[-4:-1]) / 3 - avg_epoch_loss
                epochOut += f'          avg loss speed: {avg_loss_improvement}'
                if avg_loss_improvement > 0 and avg_loss_improvement < 0.2:
                    epochs_until_0_3 = max(0, abs(avg_epoch_loss - 0.3) / avg_loss_improvement)
                    if epochs_until_0_3 > 0:
                        epochOut += f' epochs left: {epochs_until_0_3:.2f}'

            self.wait()
            self.print(epochOut)

            if self.is_main and self.checkpoint_every_epoch is not None and (
                    self.checkpoint_every_epoch == 1 or (epoch != 0 and epoch % self.checkpoint_every_epoch == 0)):
                self.save(
                    self.checkpoint_folder / f'mesh-autoencoder.ckpt.epoch_{epoch}_avg_loss_{avg_epoch_loss:.5f}_recon_{avg_recon_loss:.4f}_commit_{avg_commit_loss:.4f}.pt')

            if stop_at_loss is not None and avg_epoch_loss < stop_at_loss:
                self.print(f'Stopping training at epoch {epoch} with average loss {avg_epoch_loss}')
                if self.is_main and self.checkpoint_every_epoch is not None:
                    self.save(
                        self.checkpoint_folder / f'mesh-autoencoder.ckpt.stop_at_loss_avg_loss_{avg_epoch_loss:.3f}.pt')
                break

        self.print('Training complete')
        if display_loss_graph:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Total Loss')
            plt.plot(range(1, len(epoch_losses) + 1), epoch_recon_losses, marker='o', label='Recon Loss')
            plt.plot(range(1, len(epoch_losses) + 1), epoch_commit_losses, marker='o', label='Commit Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.grid(True)
            plt.show()
        return epoch_losses[-1]