import os
import sys
import torch
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append("VAE3D_Model")
from model_vae import VAE3D

VOXEL_DIR = "voxel_data_64"
MODEL_PATH = "VAE3D_Model/vae3d_epoch200.pth"
OUT_PATH = "CNN/latent_pairs.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 64
LATENT_SHAPE = (4, 6, 4)

model = VAE3D(latent_shape=LATENT_SHAPE, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Using model:", MODEL_PATH)

latent_dict = {}

for fname in tqdm(os.listdir(VOXEL_DIR)):
    if not fname.endswith(".npz"):
        continue

    data = np.load(os.path.join(VOXEL_DIR, fname))
    occ = data["occupancy"].astype(np.float32)[..., np.newaxis]
    flow = data["flow"].astype(np.float32)
    x_np = np.concatenate([occ, flow], axis=-1)
    x_np = np.transpose(x_np, (3, 0, 1, 2))
    x = torch.from_numpy(x_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mu, _ = model.encode(x)

    latent_dict[fname] = mu.squeeze(0).cpu()
    print(f"{fname} → mu.shape = {mu.shape}")

with open(OUT_PATH, "wb") as f:
    pickle.dump(latent_dict, f)

print(f"Saved {len(latent_dict)} latent vectors to: {OUT_PATH}")

