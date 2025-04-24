import os


obj_path = "TestSamples/examples/results/IMG_0392_inputs/IMG_0392_inputs_detail.obj"
print(os.path.exists(obj_path))  # Should print True
ply_path = "TestSamples/examples/results/IMG_0392_inputs/IMG_0392_inputs_detail.ply"

def obj_to_ply(obj_path, ply_path):
    vertices = []
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 7:
                    # Get position and RGB
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(int, parts[4:7])
                    vertices.append((x, y, z, r, g, b))
            elif line.startswith('f '):
                # Face data is optional, but if it exists, keep it
                indices = []
                parts = line.strip().split()[1:]
                for p in parts:
                    index = p.split('/')[0]  # Handle v/vt/vn or just v
                    indices.append(int(index) - 1)
                faces.append(indices)

    with open(ply_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Write vertex data
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n")

        # Write face data
        for face in faces:
            f.write(f"{len(face)} {' '.join(map(str, face))}\n")

# Example usage:
# obj_to_ply("your_model.obj", "converted_model.ply")
# ✅ Automatically run the conversion:
obj_to_ply(obj_path, ply_path)