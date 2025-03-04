import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE3D(nn.Module):
    def __init__(self, latent_shape=(4, 6, 4), latent_dim=64):
        super(VAE3D, self).__init__()
        self.latent_shape = latent_shape
        self.latent_dim = latent_dim

        # --- Encoder: 4 input channels (occupancy + flow) ---
        self.enc_conv1 = nn.Conv3d(4, 32, kernel_size=4, stride=2, padding=1)   # (32,48,32)
        self.enc_conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)  # (16,24,16)
        self.enc_conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1) # (8,12,8)
        self.enc_conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1) # (4,6,4)
        self.enc_conv5 = nn.Conv3d(256, latent_dim * 2, kernel_size=3, stride=1, padding=1)  # preserve (4,6,4)

        # --- Decoder ---
        self.dec_deconv1 = nn.ConvTranspose3d(latent_dim, 256, kernel_size=4, stride=2, padding=1)  # (8,12,8)
        self.dec_deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)         # (16,24,16)
        self.dec_deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)          # (32,48,32)
        self.dec_deconv4 = nn.ConvTranspose3d(64, 4, kernel_size=4, stride=2, padding=1)            # (64,96,64)

    def encode(self, x):
        #print(" VAE encoder received input:", x.shape)  # Debug line
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = self.enc_conv5(x)  # no activation here
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_deconv1(z))
        x = F.relu(self.dec_deconv2(x))
        x = F.relu(self.dec_deconv3(x))
        x = self.dec_deconv4(x)  # final output (64,96,64)

        occupancy_logits = x[:, 0:1]         # raw output for BCEWithLogitsLoss
        flow = torch.tanh(x[:, 1:4])         # normalize flow
        return occupancy_logits, flow

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        occ, flow = self.decode(z)
        return occ, flow, mu, logvar
