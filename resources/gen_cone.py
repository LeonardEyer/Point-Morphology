import numpy as np
from pathlib import Path

def generate_cone_pointcloud(N=50000, radius=0.5, height=1.0, filename="cone.txt"):
    """
    Generate a cone point cloud with positions and normals.
    Apex at (0,0,h/2), base at z=-h/2
    """
    points = []

    # Split points roughly between side surface and base
    n_side = int(N * 0.8)
    n_base = N - n_side

    # Side surface points
    for _ in range(n_side):
        # Random angle around axis
        theta = np.random.uniform(0, 2 * np.pi)
        # Random height along cone
        z = np.random.uniform(-height/2, height/2)
        # Corresponding radius at height
        r = radius * (height/2 - z) / height  # linear taper
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Normal: perpendicular to cone surface
        slope = radius / height
        nx = np.cos(theta)
        ny = np.sin(theta)
        nz = slope
        n = np.array([nx, nz, ny])
        n /= np.linalg.norm(n)

        points.append((x, z, y, *n))

    # Base points (flat circle at z = -h/2)
    for _ in range(n_base):
        r = np.sqrt(np.random.uniform(0, radius**2))
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = -height/2
        n = np.array([0, -1, 0])  # base normal pointing down
        points.append((x, z, y, *n))

    # Save to file
    out_path = Path(filename)
    with out_path.open("w") as f:
        f.write(f"{len(points)}\n")
        for p in points:
            f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*p))

    return out_path

# Example usage
out_file = generate_cone_pointcloud(N=500000, radius=0.5, height=1.0)
print(out_file.exists(), str(out_file))
