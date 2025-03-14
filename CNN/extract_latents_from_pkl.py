# CNN2/extract_latents_from_pkl.py
import pickle
import numpy as np

# Path to pkl file
pkl_path = "latent_pairs.pkl copy"  
npz_out_path = "latents.npz"

# Load .pkl
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} latent entries")

# Sort by image name to get consistent order
sorted_items = sorted(data.items())

# Convert to numpy array
latent_list = [v for _, v in sorted_items]
latent_array = np.stack(latent_list)  # shape: [N, 64, 4, 6, 4]

# Save as npz
np.savez(npz_out_path, latents=latent_array)
print(f" Saved {latent_array.shape} to {npz_out_path}")
