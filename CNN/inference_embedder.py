import os
import sys

sys.path.append(os.path.abspath(".."))

import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_embedder import EmbedderCNN
from VAE3D_Model.model_vae import VAE3D


def load_embedder(model_path, device):
    model = EmbedderCNN(pca_dim=343).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_vae(vae_path, device):
    model = VAE3D().to(device)
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()
    return model


def run_inference(img_path, embedder_path, pca_path, vae_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    embedder = load_embedder(embedder_path, device)
    vae = load_vae(vae_path, device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pca_pred = embedder(image_tensor).cpu().numpy()

    full_latent = pca.inverse_transform(pca_pred).reshape(1, 64, 4, 6, 4)
    latent_tensor = torch.tensor(full_latent, dtype=torch.float32).to(device)

    with torch.no_grad():
        occupancy, flow = vae.decode(latent_tensor)
        occupancy = occupancy.squeeze(0).cpu().numpy()
        flow = flow.squeeze(0).cpu().numpy()

    np.savez(output_path, occupancy=occupancy, flow=flow)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_inference(
        img_path="../example.png",
        embedder_path="embedder_cnn_best.pth",
        pca_path="pca_model.pkl",
        vae_path="../VAE3D_Model/vae3d_epoch200.pth",
        output_path="reconstructed_voxel.npz",
    )

