import numpy as np
from pathlib import Path

# Parameters
N = 500000  # total number of points
cube_size = 1.0
half = cube_size / 2

# Define faces with normals and their constant coordinate
faces = [
    ("x",  half, np.array([ 1, 0, 0])),  # +X
    ("x", -half, np.array([-1, 0, 0])),  # -X
    ("y",  half, np.array([0,  1, 0])),  # +Y
    ("y", -half, np.array([0, -1, 0])),  # -Y
    ("z",  half, np.array([0, 0,  1])),  # +Z
    ("z", -half, np.array([0, 0, -1])),  # -Z
]

points = []

for _ in range(N):
    # Pick a random face
    axis, coord, normal = faces[np.random.randint(0, 6)]
    
    # Generate random coordinates on that face
    u = np.random.uniform(-half, half)
    v = np.random.uniform(-half, half)
    
    if axis == "x":
        x = coord
        y = u
        z = v
    elif axis == "y":
        x = u
        y = coord
        z = v
    else:  # z-axis
        x = u
        y = v
        z = coord
    
    points.append((x, y, z, *normal))

# Save to file
out_path = Path("cube.txt")
with out_path.open("w") as f:
    f.write(f"{len(points)}\n")
    for p in points:
        f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*p))

out_path.exists(), str(out_path)
