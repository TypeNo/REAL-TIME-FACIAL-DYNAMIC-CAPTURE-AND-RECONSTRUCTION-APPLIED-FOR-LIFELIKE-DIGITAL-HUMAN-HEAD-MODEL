import numpy as np

file_path = './data/FLAME_texture.npz'  # Update as needed
data = np.load(file_path, allow_pickle=True, encoding='latin1')

# List all keys in the .npz file
print("Available keys in .npz file:")
print(data.files)

# Show details for each key
print("\nKey details:")
for key in data.files:
    value = data[key]
    print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
    print(f"  Values: {value}\n")
