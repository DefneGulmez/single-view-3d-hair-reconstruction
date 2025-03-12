import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# ----------------------
# Dataset Class
# ----------------------
class HairDataset(Dataset):
    def __init__(self, image_folder, latent_npz, pca_file, transform=None):
        self.image_paths = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder) if f.endswith('.png')
        ])
        self.latents = np.load(latent_npz)["latents"].reshape(len(self.image_paths), -1)
        self.pca = pickle.load(open(pca_file, "rb"))
        self.latents = self.pca.transform(self.latents)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        latent = torch.tensor(self.latents[idx], dtype=torch.float32)
        return image, latent


# ----------------------
# Embedder with IEF
# ----------------------
class EmbedderCNN(nn.Module):
    def __init__(self, pca_dim=343, steps=3):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.steps = steps
        self.fc = nn.Sequential(
            nn.Linear(2048 + pca_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, pca_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        features = self.backbone(x)  # [B, 2048]
        y = torch.zeros(B, 343).to(x.device)
        for _ in range(self.steps):
            y = y + self.fc(torch.cat([features, y], dim=1))
        return y


# ----------------------
# Training
# ----------------------
def train():
    image_folder = "../rendered_images"
    latent_npz = "latents.npz"
    pca_file = "pca_model.pkl"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor()
    ])

    dataset = HairDataset(image_folder, latent_npz, pca_file, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model = EmbedderCNN(pca_dim=343).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    print("starting")
    for epoch in range(1, 201):  # 200 epochs
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch}"):
            images, targets = images.cuda(), targets.cuda()
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch} | Loss: {epoch_loss:.6f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "embedder_cnn_best.pth")
            print(f" New best model saved (loss = {best_loss:.6f})")

        # Save periodic checkpoint
        if epoch % 50 == 0:
            ckpt_path = f"embedder_cnn_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f" Checkpoint saved: {ckpt_path}")

    print("✅ Training complete")


if __name__ == "__main__":
    train()
