import numpy as np
from pathlib import Path

# Parameters
N = 500000
sheet_x = 1.0   # width
sheet_y = 1.0   # depth
sheet_z = 0.005  # thickness — thin enough that dilation caps from both sides nearly meet

hx, hy, hz = sheet_x / 2, sheet_y / 2, sheet_z / 2

# Each face: (axis, coord, normal, u_range, v_range)
# Area-weighted so thin edges aren't over-represented
faces = [
    # Large flat faces (top/bottom)
    ("z",  hz, np.array([0, 0,  1]), (-hx, hx), (-hy, hy)),  # +Z
    ("z", -hz, np.array([0, 0, -1]), (-hx, hx), (-hy, hy)),  # -Z
    # Thin edge faces
    ("x",  hx, np.array([ 1, 0, 0]), (-hy, hy), (-hz, hz)),  # +X
    ("x", -hx, np.array([-1, 0, 0]), (-hy, hy), (-hz, hz)),  # -X
    ("y",  hy, np.array([0,  1, 0]), (-hx, hx), (-hz, hz)),  # +Y
    ("y", -hy, np.array([0, -1, 0]), (-hx, hx), (-hz, hz)),  # -Y
]

# Compute area of each face for proportional sampling
areas = np.array([
    (ur[1]-ur[0]) * (vr[1]-vr[0])
    for _, _, _, ur, vr in faces
])
probs = areas / areas.sum()
print("Face areas:", np.round(areas, 4))
print("Sampling probs:", np.round(probs, 4))

rng = np.random.default_rng(42)
face_indices = rng.choice(len(faces), size=N, p=probs)

points = []
for fi in face_indices:
    axis, coord, normal, ur, vr = faces[fi]
    u = rng.uniform(*ur)
    v = rng.uniform(*vr)

    if axis == "z":
        x, y, z = u, v, coord
    elif axis == "x":
        x, y, z = coord, u, v
    else:
        x, y, z = u, coord, v

    points.append((x, y, z, *normal))

points = np.array(points)

out_path = Path("thin-sheet.txt")
with out_path.open("w") as f:
    f.write(f"{len(points)}\n")
    for p in points:
        f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*p))

print(f"Written {len(points)} points to {out_path}")
print(f"Sheet dimensions: {sheet_x} x {sheet_y} x {sheet_z}")
print(f"Bounding box: X=[{points[:,0].min():.3f}, {points[:,0].max():.3f}]  "
      f"Y=[{points[:,1].min():.3f}, {points[:,1].max():.3f}]  "
      f"Z=[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
