import numpy as np

# === Load the .npy file ===
file_path = './data/landmark_embedding.npy'  # <- Change this to your file path
data = np.load(file_path, allow_pickle=True)

# === If data is an object array containing a dict ===
if isinstance(data, np.ndarray) and data.dtype == 'object':
    try:
        data = data.item()
    except ValueError:
        print("Unable to unpack object array. Showing first element instead.")
        data = data[0]

# === If data is a dictionary ===
if isinstance(data, dict):
    print("Keys and shapes:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
        else:
            print(f"{key}: type = {type(value)} (not ndarray)")
else:
    print("Data is not a dictionary. Type:", type(data))
