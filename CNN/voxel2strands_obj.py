import numpy as np
import os

def load_voxel_data(npz_path):
    data = np.load(npz_path)
    occupancy = data['occupancy']  # shape: (D, H, W)
    flow = data['flow']            # shape: (3, D, H, W)
    return occupancy, flow

def trace_strands(occupancy, flow, threshold=0.5, max_steps=20, step_size=1.0):
    if occupancy.ndim == 4:
        occupancy = occupancy.squeeze(0)
    D, H, W = occupancy.shape
    strands = []

    for z in range(D):
        for y in range(H):
            for x in range(W):
                if occupancy[z, y, x] > threshold:
                    strand = []
                    pos = np.array([x, y, z], dtype=np.float32)

                    for _ in range(max_steps):
                        strand.append(pos.copy())

                        # Clamp coordinates
                        ix = int(np.clip(pos[0], 0, W - 1))
                        iy = int(np.clip(pos[1], 0, H - 1))
                        iz = int(np.clip(pos[2], 0, D - 1))

                        # Correct indexing: flow[:, z, y, x]
                        direction = flow[:, iz, iy, ix]

                        if np.linalg.norm(direction) < 1e-3:
                            break

                        pos += direction * step_size

                        if not (0 <= pos[0] < W and 0 <= pos[1] < H and 0 <= pos[2] < D):
                            break

                    if len(strand) >= 2:
                        strands.append(strand)

    return strands

def write_obj(strands, out_path, scale=0.01):
    with open(out_path, 'w') as f:
        v_count = 1
        for strand in strands:
            for pt in strand:
                x, y, z = pt * scale
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
            for i in range(len(strand) - 1):
                f.write(f"l {v_count + i} {v_count + i + 1}\n")
            v_count += len(strand)
    print(f"✅ Saved {len(strands)} strands to: {out_path}")

if __name__ == "__main__":
    input_npz = "/Users/defnegulmez/Desktop/IG3D_Project/animation/ages/hair_teen.npz"  
    output_obj = "/Users/defnegulmez/Desktop/IG3D_Project/animation/ages/hair_teen.obj"  

    print(" Loading voxel data...")
    occ, flow = load_voxel_data(input_npz)

    print("Tracing strands...")
    strands = trace_strands(occ, flow, threshold=0.2, max_steps=30, step_size=1.0)

    print("Exporting to OBJ...")
    write_obj(strands, output_obj)
