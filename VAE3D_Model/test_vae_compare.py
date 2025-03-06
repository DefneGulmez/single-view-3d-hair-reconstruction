import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model_vae import VAE3D

# --- Paths ---
VOXEL_PATH = "../example.npz"  #Path for testing file
MODEL_PATH = "vae3d_epoch200.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = VAE3D().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Load Original Data ---
data = np.load(VOXEL_PATH)
occ_np = data["occupancy"].astype(np.float32)  # (D,H,W)
flow_np = data["flow"].astype(np.float32)      # (D,H,W,3)

# Prepare input tensor
occ_exp = occ_np[..., np.newaxis]  # (D,H,W,1)
x_np = np.concatenate([occ_exp, flow_np], axis=-1)  # (D,H,W,4)
x_np = np.transpose(x_np, (3, 0, 1, 2))  # (4,D,H,W)
x = torch.from_numpy(x_np).unsqueeze(0).to(DEVICE)

# --- Run Model ---
with torch.no_grad():
    recon_occ_logits, recon_flow, *_ = model(x)
    recon_occ = torch.sigmoid(recon_occ_logits)

# --- Extract slices for comparison ---
slice_z = occ_np.shape[2] // 2  # mid Z-slice

# Occupancy (original + recon)
occ_orig_slice = occ_np[:, :, slice_z]
occ_recon_slice = recon_occ[0, 0, :, :, slice_z].cpu().numpy()

# Flow field (XY projection)
flow_orig = np.transpose(flow_np, (3, 0, 1, 2))  # (3,D,H,W)
flow_recon = recon_flow[0].cpu().numpy()         # (3,D,H,W)
step = 4  # downsample for quiver plot

U_orig = flow_orig[0, ::step, ::step, slice_z]
V_orig = flow_orig[1, ::step, ::step, slice_z]
U_recon = flow_recon[0, ::step, ::step, slice_z]
V_recon = flow_recon[1, ::step, ::step, slice_z]

# --- Plot ---
plt.figure(figsize=(12, 8))

# Occupancy comparison
plt.subplot(2, 2, 1)
plt.title("Original Occupancy (Z mid)")
plt.imshow(occ_orig_slice.T, origin="lower", cmap="gray")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title("Reconstructed Occupancy (Z mid)")
plt.imshow(occ_recon_slice.T, origin="lower", cmap="gray")
plt.colorbar()

# Flow field comparison
plt.subplot(2, 2, 3)
plt.title("Original Flow Field (XY @ Z mid)")
plt.quiver(U_orig.T, V_orig.T)
plt.axis("equal")

plt.subplot(2, 2, 4)
plt.title("Reconstructed Flow Field (XY @ Z mid)")
plt.quiver(U_recon.T, V_recon.T)
plt.axis("equal")

plt.tight_layout()
plt.show()
