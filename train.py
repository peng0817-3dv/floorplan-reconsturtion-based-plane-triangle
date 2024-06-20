import os

from plane_tri_meshgpt.planetri_dataset import PlaneTriDataset
from utils.load_shp import load_filename
from pathlib import Path


DATA_RESOURCE_PATH = 'G:\workspace_plane2DDL\label_shp_root'

project_name = 'demo'
project_working_dir = f'./{project_name}'
project_working_dir = Path(project_working_dir)
project_working_dir.mkdir(exist_ok=True, parents=True)
dataset_path = project_working_dir / (project_name + ".npz")

if not os.path.isfile(dataset_path):
    data = load_filename(DATA_RESOURCE_PATH,variations=20)
    dataset = PlaneTriDataset(data)
    dataset.generate_face_edges()
    dataset.save(dataset_path)

dataset = PlaneTriDataset.load(dataset_path)
print(dataset.data[0].keys())


