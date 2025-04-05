import torch
from torch.hub import load_state_dict_from_url
from face_alignment.utils import *

# URL of the model
model_url = "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip"

# Load the model state_dict
net =  torch.jit.load(load_file_from_url(model_url))

# Print model architecture
print(net)
