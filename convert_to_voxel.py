import os
import numpy as np
import struct
import array
from tqdm import tqdm

# Grid resolution and uniform padding used to normalize strands into the voxel grid
GRID_DIMS = (64, 96, 64)
PADDING = 0.05

def load_hair_data(filepath):
    """Load .data strand file: returns list of strands, each strand = list of [x,y,z] points."""
    strands = []
    with open(filepath, "rb") as f:
        num_strands = struct.unpack('<i', f.read(4))[0]
        for _ in range(num_strands):
            num_verts = struct.unpack('<i', f.read(4))[0]
            verts = array.array('f')
            verts.fromfile(f, 3 * num_verts)
            if num_verts < 2:
                continue
            strand = []
            for i in range(num_verts):
                x = verts[3 * i]
                y = verts[3 * i + 1]
                z = verts[3 * i + 2]
                strand.append([x, y, z])
            strands.append(strand)
    return strands

def voxelize_strands(strands, grid_dims=GRID_DIMS, padding=PADDING):
    """Voxelize strands into occupancy (O) and averaged orientation field (F)."""
    all_points = np.array([pt for strand in strands for pt in strand])
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    center = (min_coords + max_coords) / 2
    scale = (1 + padding) * np.max(max_coords - min_coords)

    O = np.zeros(grid_dims, dtype=np.uint8)
    F = np.zeros(grid_dims + (3,), dtype=np.float32)
    counts = np.zeros(grid_dims, dtype=np.int32)

    def world_to_voxel(pt):
        normed = (pt - center) / scale + 0.5
        voxel = (normed * grid_dims).astype(int)
        return np.clip(voxel, 0, np.array(grid_dims) - 1)

    for strand in strands:
        for i in range(len(strand) - 1):
            p1 = np.array(strand[i])
            p2 = np.array(strand[i + 1])
            direction = p2 - p1
            n = np.linalg.norm(direction)
            if n == 0:
                continue
            direction /= n
            voxel_idx = tuple(world_to_voxel(p1))
            O[voxel_idx] = 1
            F[voxel_idx] += direction
            counts[voxel_idx] += 1

    nonzero = counts > 0
    F[nonzero] = F[nonzero] / counts[nonzero][..., None]
    return O, F

def convert_strands_to_voxel(data_path, out_path):
    strands = load_hair_data(data_path)
    O, F = voxelize_strands(strands)
    np.savez_compressed(out_path, occupancy=O, flow=F)
    print(f"Saved voxel data: {out_path}")

def batch_convert_strands_to_voxel(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".data")]
    for filename in tqdm(files, desc="Converting .data to voxel"):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + "_voxel.npz"
        output_path = os.path.join(output_dir, output_filename)
        try:
            convert_strands_to_voxel(input_path, output_path)
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    batch_convert_strands_to_voxel(
        input_dir="/Users/defnegulmez/Desktop/IG3D_Project/hairstyles",
        output_dir="/Users/defnegulmez/Desktop/IG3D_Project/voxel_data"
    )
