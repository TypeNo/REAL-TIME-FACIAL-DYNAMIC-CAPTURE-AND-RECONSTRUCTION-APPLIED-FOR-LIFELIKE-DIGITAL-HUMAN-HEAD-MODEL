import torch
from torch.hub import load_state_dict_from_url
from face_alignment.utils import *
from decalib.utils.config import cfg
import matplotlib.pyplot as plt

model = torch.load (cfg.pretrained_modelpath)
#model.eval()
#print(model.keys())
#for key, weight in model["E_flame"].items():
#    print(f"Layer: {key}, Shape: {weight.shape}")

#weights = model["E_flame"]["encoder.conv1.weight"].cpu().detach().numpy()
#plt.imshow(weights[0, 0], cmap="gray")  # Visualizing the first filter of the first channel
#plt.show()

print("Epoch:", model["epoch"])
print("Iteration:", model["iter"])
print("Batch size:", model["batch_size"])




