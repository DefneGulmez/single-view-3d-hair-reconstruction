import os
import numpy as np
import torch
from torch.utils.data import Dataset

class VoxelHairDataset(Dataset):
    def __init__(self, voxel_dir, transform=None):
        """
        Args:
            voxel_dir (str): path to the directory containing .npz voxel files
        """
        self.voxel_dir = voxel_dir
        self.voxel_files = sorted([
            f for f in os.listdir(voxel_dir)
            if f.endswith('.npz')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.voxel_dir, self.voxel_files[idx])

        try:
            data = np.load(npz_path)
            occupancy = data['occupancy'].astype(np.float32)
            flow = data['flow'].astype(np.float32)

            if idx == 0:
                print(f" Loaded {npz_path}")
                print(f" Occupancy shape: {occupancy.shape}")
                print(f" Flow shape: {flow.shape}")

        except Exception as e:
            print(f" Failed to lood {npz_path}: {e}")
            raise e

        # Combine occupancy and flow → 4 channels
        occ = occupancy[..., np.newaxis]                  # (D,H,W,1)
        input_voxel = np.concatenate([occ, flow], axis=-1)  # (D,H,W,4)
        input_voxel = np.transpose(input_voxel, (3, 0, 1, 2))  # (4,D,H,W)

        input_tensor = torch.from_numpy(input_voxel)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, self.voxel_files[idx]
