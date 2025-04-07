import pickle
import numpy as np

try:
    import torch
except ImportError:
    torch = None

def load_pkl(path):
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f, encoding='latin1')  # Compatibility encoding
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None
    return data

def print_data_info(obj, name='root', indent=0):
    pad = '  ' * indent
    shape = None
    values = None

    # Handle PyTorch tensors
    if torch and isinstance(obj, torch.Tensor):
        shape = tuple(obj.shape)
        values = obj.detach().cpu().numpy()
    # Handle NumPy arrays
    elif isinstance(obj, np.ndarray):
        shape = obj.shape
        values = obj
    # Handle basic types
    elif isinstance(obj, (int, float, str)):
        print(f"{pad}{name}: {type(obj).__name__} = {repr(obj)}")
        return
    # Handle lists
    elif isinstance(obj, list):
        print(f"{pad}{name}: list of length {len(obj)}")
        for i, item in enumerate(obj[:5]):  # Show only first 5
            print_data_info(item, name=f"{name}[{i}]", indent=indent + 1)
        return
    # Handle dicts
    elif isinstance(obj, dict):
        print(f"{pad}{name}: dict with {len(obj)} keys")
        for k, v in list(obj.items())[:10]:  # Show only first 10
            print_data_info(v, name=f"{name}[{repr(k)}]", indent=indent + 1)
        return
    else:
        print(f"{pad}{name}: {type(obj).__name__}")
        return

    # Print arrays/tensors
    print(f"{pad}{name}: shape = {shape}")
    print(f"{pad}  values = {values.flatten()[:5]}{'...' if values.size > 5 else ''}")

# === MAIN ===
path = './data/generic_model.pkl'
data = load_pkl(path)
if data is not None:
    print("✅ File loaded successfully!\n")
    print_data_info(data)
