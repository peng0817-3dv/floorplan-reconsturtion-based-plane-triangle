import os

from plane_tri_meshgpt.planetri_dataset import PlaneTriDataset
from utils.load_shp import load_filename
from pathlib import Path
from plane_tri_meshgpt.meshgpt_pytorch import MeshAutoencoder
from plane_tri_meshgpt.trainer import MeshAutoencoderTrainer

DATA_RESOURCE_PATH = 'G:\workspace_plane2DDL\label_shp_root'

project_name = 'demo'
project_working_dir = f'./{project_name}'
project_working_dir = Path(project_working_dir)
project_working_dir.mkdir(exist_ok=True, parents=True)
dataset_path = project_working_dir / (project_name + ".npz")

if not os.path.isfile(dataset_path):
    data = load_filename(DATA_RESOURCE_PATH)
    dataset = PlaneTriDataset(data)
    dataset.generate_face_edges()
    dataset.save(dataset_path)

dataset = PlaneTriDataset.load(dataset_path)
print(dataset.data[0].keys())


# 训练
autoencoder = MeshAutoencoder(
    decoder_dims_through_depth=(128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
    codebook_size=2048,
    # Smaller vocab size will speed up the transformer training, however if you are training on meshes more then 250 triangle, I'd advice to use 16384 codebook size
    dim_codebook=192,
    dim_area_embed=16,
    dim_coor_embed=16,
    dim_normal_embed=16,
    dim_angle_embed=8,

    attn_decoder_depth=4,
    attn_encoder_depth=2
).to("cuda")
total_params = sum(p.numel() for p in autoencoder.parameters())
total_params = f"{total_params / 1000000:.1f}M"
print(f"Total parameters: {total_params}")

# Have at least 400-2000 items in the dataset, use this to multiply the dataset
dataset.data = [dict(d) for d in dataset.data] * 10
print(len(dataset.data))

batch_size=16 # The batch size should be max 64.
grad_accum_every = 4
# So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
learning_rate = 1e-3 # Start with 1e-3 then at staggnation around 0.35, you can lower it to 1e-4.

autoencoder.commit_loss_weight = 0.2 # Set dependant on the dataset size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                             batch_size=batch_size,
                                             grad_accum_every = grad_accum_every,
                                             learning_rate = learning_rate,
                                             checkpoint_every_epoch=40)

continue_check_file = 'checkpoints/mesh-autoencoder.ckpt.epoch_40_avg_loss_0.57972_recon_0.5555_commit_0.1210.pt'
if os.path.exists(continue_check_file):
    autoencoder_trainer.load(continue_check_file)
# 训练480个epoch
loss = autoencoder_trainer.train(480,stop_at_loss = 0.2, display_loss_graph= True,)
# 训练完后的权值存放为encoder.pt 权重
autoencoder_trainer.save(f'{project_working_dir}\mesh-encoder_{project_name}.pt')
