import pickle
import pprint
import numpy as np

filename = './data/generic_model.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

print(f"Type of data: {type(data)}\n")

def print_with_shapes(obj, indent=0):
    prefix = ' ' * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}: ", end="")
            print_with_shapes(v, indent + 4)
    elif isinstance(obj, list):
        print(f"List of length {len(obj)}")
        if len(obj) > 0:
            print_with_shapes(obj[0], indent + 4)
    elif isinstance(obj, np.ndarray):
        print(f"np.ndarray with shape {obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{type(obj).__name__} - {str(obj)[:60]}")

print("Contents of the pickle file:\n")
print_with_shapes(data)

# Check the shape of 'shapedirs'
shapedirs = data.get('shapedirs', None)

if shapedirs is not None:
    if isinstance(shapedirs, np.ndarray):
        print(f"shapedirs is a NumPy array with shape: {shapedirs.shape}")
    elif hasattr(shapedirs, 'shape'):
        print(f"shapedirs shape (from non-ndarray): {shapedirs.shape}")
    else:
        print("shapedirs exists but has unknown structure.")
else:
    print("shapedirs key not found in the model data.")
