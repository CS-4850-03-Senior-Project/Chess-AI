import matplotlib.pyplot as plt
import numpy as np
from EvaluationNetworkTanh import EvaluationNetwork
import torch

model_checkpoint = torch.load("MODELS/RedChessAI20241005164331.pth", weights_only=True)
model = EvaluationNetwork()
model.load_state_dict(model_checkpoint['model_state_dict'])

# Convert the tensor to a numpy array
weights = model.conv2.weight
print(weights.shape)
filter_count = weights.shape[0]
weights = weights.detach().cpu().numpy()


# Normalize weights for visualization (optional)
min_weight = np.min(weights)
max_weight = np.max(weights)
weights = (weights - min_weight) / (max_weight - min_weight)

fig, axes = plt.subplots(4, 8, figsize=(10, 5))
axes = axes.flatten()

# Plot each filter
for i in range(filter_count):
    filter_weights = weights[i,0]
    axes[i].imshow(filter_weights, cmap='gnuplot')
    axes[i].axis('off')

plt.show()
