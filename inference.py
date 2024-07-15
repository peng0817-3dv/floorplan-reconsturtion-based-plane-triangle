import os

import torch
import random
from tqdm import tqdm
from plane_tri_meshgpt import mesh_render

from pathlib import Path
from plane_tri_meshgpt.planetri_dataset import PlaneTriDataset
from plane_tri_meshgpt.meshgpt_pytorch import MeshAutoencoder
from utils.load_shp import load_filename

from packaging import version
from meshgpt_pytorch.version import __version__

# 载入数据集

project_name = "demo"
DATA_RESOURCE_PATH = 'G:\workspace_plane2DDL\label_shp_root'
working_dir = f'./{project_name}'
# 工作目录
working_dir = Path(working_dir)
working_dir.mkdir(exist_ok=True, parents=True)

# 数据集目录
dataset_path = 'I:\RECORD/7_12_360epoch_1000item/demo.npz'

if not os.path.isfile(dataset_path):
    data = load_filename(DATA_RESOURCE_PATH)
    dataset = PlaneTriDataset(data)
    dataset.generate_face_edges()
    dataset.save(dataset_path)

dataset = PlaneTriDataset.load(dataset_path)
print(dataset.data[0].keys())

# 初始化一个模型
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

# 将trainer存放的checkpoint中的模型权重载入模型
# 这部分实际上是拆解了trainer的load函数中的关于模型载入的代码
path = 'I:\RECORD/7_12_360epoch_1000item/mesh-autoencoder.ckpt.stop_at_loss_avg_loss_0.199.pt'
path = Path(path)
pkg = torch.load(str(path))
if version.parse(__version__) != version.parse(pkg['version']):
    print(
        f'loading saved mesh autoencoder at version {pkg["version"]}, but current package version is {__version__}')
autoencoder.load_state_dict(pkg['model'])

min_mse, max_mse = float('inf'), float('-inf')
min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
random_samples, random_samples_pred, all_random_samples = [], [], []
ori_samples, pre_samples = [],[]
total_mse = 0.0
sample_size = 30


#采样dataset中的前{sample_size}个网格
for item in tqdm(dataset.data[:sample_size]):
    # 利用训练好的autoencoder对dataset中的网格推理生成token code
    codes = autoencoder.tokenize(vertices=item['vertices'], faces=item['faces'], face_edges=item['face_edges'])
    codes = codes.flatten().unsqueeze(0)
    codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers]

    # 将得到的token code解码回mesh
    coords, mask = autoencoder.decode_from_codes_to_faces(codes)
    # orgs 用于记录原来的顶点集
    orgs = item['vertices'][item['faces']].unsqueeze(0)

    # 计算解码误差：方差
    mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu()) ** 2)
    # 累计总误差
    total_mse += mse

    # 记录最小解码误差和最大解码误差
    if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
    if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs

    ori_samples.append(orgs)
    pre_samples.append(coords)

# 终端输出推断误差
print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')

for i,ori in enumerate(ori_samples) :
    pre = pre_samples[i]
    root_dir = os.path.join(f'{working_dir}','mesh',str(i))
    mesh_render.save_mesh_pair(root_dir=root_dir,mesh_pair=[ori,pre])

# 渲染解码的网格可视化
# mesh_render.combind_mesh_with_rows(f'{working_dir}/view', all_random_samples)