from torch.utils.data import Dataset
import numpy as np
from plane_tri_meshgpt.data import derive_face_edges_from_faces

class PlaneTriDataset(Dataset):
    """
    用于加载和处理平面网格数据的PyTorch数据集。
    “PlaneTriDataset”提供了从文件加载网格数据、嵌入文本信息、生成面边缘和生成代码的功能。

    属性：
        data（list）：网格数据条目的列表。每个条目都是一个包含以下键的字典：
            顶点（torch. Tensor）：具有形状（num_vertices，3）的顶点张量。
            面（torch. Tensor）：具有形状（num_faces，3）的面的张量。
            面特征（torch. Tensor）：具有形状(numm_faces,7) 的面特征张量。
            face_edges张量：带有形状（num_faces，num_edges）的面边缘张量。


    示例用法：


    “”
    data=[
        {'顶点'：torch. tenor（[[1,2,3]，[4,5,6]]，dtype=torch.float32），'面'：torch.tenor（[[0,1,2]]，dtype=torch.long），'面特征'：...}，
        {'顶点'：torch. tenor（[[10,20,30]，[40,50,60]]，dtype=torch.float32），'面'：torch.tenor（[[1,2,0]]，dtype=torch.long），'面特征'：...}，
    ]


    #创建MeshDataset实例
    mesh_dataset数据集


    #将MeshDataset保存到磁盘
    mesh_dataset.save（'mesh_dataset. npz'）


    #从磁盘加载MeshDataset
    loaded_mesh_dataset=MeshDataset. load（'mesh_dataset.npz'）

    #生成面的拓扑边缘，这样就不需要在训练期间每次都这样做
    数据集.generate_face_edges（）
    """
    def __init__(self, data):
        self.data = data
        print(f"[PlaneTriDataset] Created from {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    def save(self, path):
        np.savez_compressed(path, self.data, allow_pickle=True)
        print(f"[PlaneTriDataset] Saved {len(self.data)} entries at {path}")

    @classmethod
    def load(cls, path):
        loaded_data = np.load(path, allow_pickle=True)
        data = []
        for item in loaded_data["arr_0"]:
            data.append(item)
        print(f"[PlaneTriDataset] Loaded {len(data)} entries")
        return cls(data)

    def sort_dataset_keys(self):
        desired_order = ['vertices', 'faces', 'faces_feature','face_edges']
        self.data = [
            {key: d[key] for key in desired_order if key in d} for d in self.data
        ]

    def generate_face_edges(self):
        i = 0
        for item in self.data:
            if 'face_edges' not in item:
                item['face_edges'] = derive_face_edges_from_faces(item['faces'])
                i += 1

        self.sort_dataset_keys()
        print(f"[MeshDataset] Generated face_edges for {i}/{len(self.data)} entries")