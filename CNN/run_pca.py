# CNN2/run_pca.py
import numpy as np
import pickle
from sklearn.decomposition import PCA

# Input and output paths
latent_path = "latents.npz"
pca_out_path = "pca_model.pkl"

# Load latents
latents = np.load(latent_path)["latents"]  # [N, 64, 4, 6, 4]
N = latents.shape[0]

# Flatten each latent vector
latents_flat = latents.reshape(N, -1)  # [N, 6144]

# Fit PCA
pca = PCA(n_components=343)
pca.fit(latents_flat)

# Save PCA model
with open(pca_out_path, "wb") as f:
    pickle.dump(pca, f)

print(f"✅ PCA trained on {N} samples.")
print(f"Saved PCA model to {pca_out_path}")
