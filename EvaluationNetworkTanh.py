import torch.nn as nn

# Changing this hyperparameter changes the model (requires retraining)
hidden_size = 128
intermediate_layers = 2

# Neural network model to predict the value of a board state
class EvaluationNetwork(nn.Module):
    def __init__(self):
        super(EvaluationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_size)
        self.intermediate_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) 
             for _ in range(intermediate_layers)])
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = self.tanh(self.fc1(x))
        for layer in self.intermediate_layers:
            x = self.tanh(layer(x))
        x = self.fc2(x)
        return x