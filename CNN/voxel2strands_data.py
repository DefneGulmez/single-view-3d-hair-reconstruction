import numpy as np
import struct
from tqdm import tqdm

#paths to input and where I want the output
INPUT_NPZ = "reconstructed_voxel.npz"
OUTPUT_DATA = "reconstructed_voxel.data"

def grow_strands(occupancy, flow, step=0.5, max_length=100, scalp_z_threshold=64):
    print("Occupancy shape:", occupancy.shape)
    print("flow shape:", flow.shape)
    
    # Only squeeze occupancy
    if occupancy.ndim == 4 and occupancy.shape[0] == 1:
        occupancy = occupancy.squeeze(0)

    res_z, res_y, res_x = occupancy.shape
    strands = []

    for z in tqdm(range(res_z)):
        for y in range(res_y):
            for x in range(res_x):
                if occupancy[z, y, x] > 0.2 and z <= scalp_z_threshold:
                    strand = []
                    pos = np.array([x + 0.5, y + 0.5, z + 0.5])
                    for _ in range(max_length):
                        strand.append(pos.copy())
                        gx, gy, gz = np.clip(pos.astype(int), 0, [res_x-1, res_y-1, res_z-1])
                        direction = flow[:, gz, gy, gx]
                        norm = np.linalg.norm(direction)
                        if norm < 1e-3:
                            break
                        direction /= norm
                        next_pos = pos + step * direction
                        if (
                            next_pos[0] < 0 or next_pos[0] >= res_x or
                            next_pos[1] < 0 or next_pos[1] >= res_y or
                            next_pos[2] < 0 or next_pos[2] >= res_z or
                            occupancy[int(next_pos[2]), int(next_pos[1]), int(next_pos[0])] < 0.5
                        ):
                            break
                        pos = next_pos
                    if len(strand) >= 5:
                        strands.append(np.array(strand))
    return strands

def save_strands_to_data(strands, output_path):
    with open(output_path, 'wb') as f:
        f.write(struct.pack('i', len(strands)))
        for strand in strands:
            f.write(struct.pack('i', len(strand)))
            for v in strand:
                f.write(struct.pack('3f', *v))

if __name__ == "__main__":
    print(f"Loading: {INPUT_NPZ}")
    data = np.load(INPUT_NPZ)
    occupancy = data["occupancy"]
    flow = data["flow"]

    print("Growing strands...")
    strands = grow_strands(occupancy, flow)

    print(f"Saving {len(strands)} strands to: {OUTPUT_DATA}")
    save_strands_to_data(strands, OUTPUT_DATA)
    print("Done.")
