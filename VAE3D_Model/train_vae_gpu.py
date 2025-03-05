import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import autocast, GradScaler

from dataset_voxels import VoxelHairDataset
from model_vae import VAE3D

# --- Training settings ---
BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- AMP for CUDA ---
scaler = GradScaler()
print(" AMP and GPU enabled")
print(" Starting training...")

# --- Loss functions ---
bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

def vae_loss_function(recon_occ_logits, recon_flow, true_occ, true_flow, mu, logvar):
    bce = bce_loss(recon_occ_logits, true_occ)
    mask = (true_occ > 0.5).expand_as(true_flow)
    l1 = l1_loss(recon_flow[mask], true_flow[mask]) if mask.any() else torch.tensor(0.0, device=recon_occ_logits.device)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + l1 + kl, (bce.item(), l1.item(), kl.item())

def train():
    voxel_data_path = "/content/drive/MyDrive/IG3D_Project/voxel_data_64"
    dataset = VoxelHairDataset(voxel_data_path)
    print(f"📦 Loaded dataset with {len(dataset)} samples from {voxel_data_path}")

    if len(dataset) == 0:
        print("❌ ERROR: No .npz files found. Aborting.") #emojies for better visibility in the outputs
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = VAE3D().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        total_loss, total_bce, total_l1, total_kl = 0, 0, 0, 0

        for batch, _ in dataloader:
            batch = batch.to(DEVICE)
            true_occ = batch[:, 0:1]
            true_flow = batch[:, 1:4]

            with autocast("cuda"):
                recon_occ_logits, recon_flow, mu, logvar = model(batch)
                loss, (bce, l1, kl) = vae_loss_function(recon_occ_logits, recon_flow, true_occ, true_flow, mu, logvar)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_bce += bce
            total_l1 += l1
            total_kl += kl

        duration = time.time() - start_time
        print(f"[Epoch {epoch+1:03d}] Time: {duration:.2f}s")

        if (epoch + 1) % 50 == 0 or epoch == NUM_EPOCHS - 1:
            save_path = f"vae3d_epoch{epoch+1:03d}.pth"
            torch.save(model.state_dict(), save_path)
            print(f" Saved model to {save_path}")

if __name__ == "__main__":
    train()