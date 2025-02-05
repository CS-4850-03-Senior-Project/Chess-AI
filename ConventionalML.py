# Here I am going to attempt to pretrain a neural network on Stockfish 
#   evaluations. This should give a model that makes reasonable moves quickly
#   and then we can tune it with reinforcement learning

# The data has a FEN string representation of the board and the stockfish
#   evaluation. The evaluation appears to be in different formats:
#   - a centipawn score
#   - a mate score
# https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations/data  

# Proposed Solution
# 1. Load EvaluationNetwork (same as the reinforcement learning experiment) 
# 2. Preprocess scores
#   - Encode mate scores in a continuous space
#   - Normalize score data
# 3. Training loop
#   - Convert FEN string to chess.Board
#   - Convert board to tensor (algorithm is in RL file; board_to_tensor())
# 4. Save
#   - Model weights and optimizer state 

import torch
import torch.nn as nn
import torch.optim as optim
import chess
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from BoardToTensor import board_to_tensor
from EvaluationNetworkTanh import EvaluationNetwork
from datetime import datetime
import os

def preprocess_score(score_str):
  score_str = str(score_str).strip()
  if score_str.startswith('#'):
    # Mate scores in the form #-5
    mate_in = int(score_str[1:])
    if mate_in > 0:
      score = 10_000 - mate_in * 100
    else:
      score = -10_000 + mate_in * 100
  else:
    # Centipawn scores
    score = float(score_str)

  # Normalishization
  score /= 1000.0

  return score

class ChessDataset(Dataset):
  def __init__(self, csv_file):
    self.data = pd.read_csv(csv_file, header=0)
    self.length = len(self.data)

  def __len__(self):
    return self.length
  
  def __getitem__(self, index):
    fen = self.data.iloc[index]['FEN']
    score_str = self.data.iloc[index]['Evaluation']

    board = chess.Board(fen)
    board_tensor = board_to_tensor(board)

    score = preprocess_score(score_str)

    return board_tensor, torch.tensor([score], dtype=torch.float32)
  
# Hyperparameters
batch_size = 64
num_epochs = 10
learning_rate = 0.0001

# Check for CUDA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
dataset = ChessDataset('./ChessEvaluations/chessData.csv')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = EvaluationNetwork().to(device)
model.train()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Checkpoint details
start_epoch = 0
load_checkpoint = True
checkpoint_filepath = "MODELS/Pretrained20241009024418.pth"

# Load checkpoint if desired
if load_checkpoint:
    checkpoint = torch.load(checkpoint_filepath)
    # TODO: Clear out this manual epoch overwrite after training resumes
    start_epoch = 5 # checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Loaded checkpoint "{checkpoint_filepath}" at epoch {start_epoch}')


# Training loop
report_on_batch = 2000 # How often to print progress reports
for epoch in range(start_epoch, num_epochs):
  running_loss = 0.0
  for i, (inputs, targets) in enumerate(dataloader):
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print statistics
    running_loss += loss.item()
    if (i + 1) % report_on_batch == 0:
      print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}], Loss: {running_loss / report_on_batch:.2f}')
      running_loss = 0.0

  # Save model weights and optimizer state every epoch
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  models_dir = "MODELS"
  model_filename = f"Pretrained{timestamp}.pth"
  model_path = os.path.join(models_dir,model_filename)
  torch.save({
      'epoch': epoch + 1,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
  }, model_path)