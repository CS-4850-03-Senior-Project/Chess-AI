import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import os
from datetime import datetime


# Hyperparameters
learning_rate = 0.0001
num_episodes = 100
initial_temperature = 0.3
temperature_decay = 0.995
min_temperature = 0.05

# Reward Parameters
reward_decay = 1.0

# Changing this hyperparameter changes the model (requires retraining)
hidden_size = 128
intermediate_layers = 2

# Endgame positions (We will focus training on endgame first)
endgame_training = True
endgame_positions = [
    # Use Forsyth-Edwards Notation strings
]

# Create MODELS directory if it doesn't exist
models_dir = "MODELS"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Device configuration (I have multiple GPUs; I want to use device 0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to convert the chess board state to a tensor representation
def board_to_tensor(board):
    # Initialize a tensor of shape (14, 8, 8) filled with zeros
    board_tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Mapping from piece type to index in the tensor channels
    piece_indices = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # Map pieces on the board to the tensor
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Determine the offset for the piece color
            color_offset = 0 if piece.color == chess.WHITE else 6
            # Get the index for the piece type
            piece_index = piece_indices[piece.piece_type] + color_offset
            # Convert the square index to 2D coordinates (x, y)
            x, y = divmod(square, 8)
            # Set the presence of the piece in the tensor
            board_tensor[piece_index, x, y] = 1.0
    
    # Initialize attack maps for white and black pieces
    white_attacks = np.zeros((8, 8), dtype=np.float32)
    black_attacks = np.zeros((8, 8), dtype=np.float32)
    
    # Generate attack maps
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Get the squares attacked by the piece
            attacks = board.attacks(square)
            for attacked_square in attacks:
                x, y = divmod(attacked_square, 8)
                if piece.color == chess.WHITE:
                    white_attacks[x, y] = 1.0
                else:
                    black_attacks[x, y] = 1.0
    
    # Add attack maps to the tensor
    board_tensor[12] = white_attacks
    board_tensor[13] = black_attacks
    
    # Convert the numpy array to a PyTorch tensor and add batch dimension
    return torch.from_numpy(board_tensor).unsqueeze(0).to(device)

# Neural network model to predict the value of a board state
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_size)
        self.fci = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        for _ in range(intermediate_layers):
            x = self.relu(self.fci(x))
        x = self.fc2(x)
        return x

# Initialize the value network and optimizer
trainee_model = ValueNetwork().to(device)
optimizer = optim.Adam(trainee_model.parameters(), lr=learning_rate)
opponent_model = ValueNetwork().to(device)

# Starting point of trainee model
load_trainee = True
trainee_model_filename = "RedChessAI20241001194738.pth" 

# Model of the opponent (can be the same as trainee)
load_opponent = True # If false, the opponent will play randomly
opponent_model_load_file = "RedChessAI20241001194738.pth" 

# Load trainee
if load_trainee:
  model_path = os.path.join(models_dir, trainee_model_filename)
  trainee_model.load_state_dict(torch.load(model_path, weights_only=True))
trainee_model.to(device)

# Load opponent
if load_opponent:
  opponent_model_path = os.path.join(models_dir, opponent_model_load_file)
  opponent_model.load_state_dict(
      torch.load(opponent_model_path, weights_only=True)) 

# Training loop
temperature = initial_temperature
total_reward = 0.0
for episode in range(num_episodes):
    board = chess.Board()
    states = []
    rewards = []
    
    # Randomly decide if the AI plays as White or Black
    ai_plays_white = np.random.choice([True, False])
    
    # Count the number of turns in the game
    turn_count = 0

    while not board.is_game_over():
        turn_count += 1

        if board.turn == (chess.WHITE if ai_plays_white else chess.BLACK):
            # Trainee's turn
            legal_moves = list(board.legal_moves)
            move_values = []

            # Evaluate all legal moves using current_model
            for move in legal_moves:
                board.push(move)
                state_tensor = board_to_tensor(board)
                value = trainee_model(state_tensor).item()
                move_values.append(value)
                board.pop()

            # Apply softmax with temperature to the predicted values
            logits = np.array(move_values) / temperature
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()

            # Select a move based on the probabilities
            move_index = np.random.choice(len(legal_moves), p=probabilities)
            chosen_move = legal_moves[move_index]

            # Make the chosen move
            board.push(chosen_move)
            # Store the state after the move
            states.append(board_to_tensor(board))
        else:
            # Opponent's turn
            legal_moves = list(board.legal_moves)

            if load_opponent:
              move_values = []
              # Evaluate all legal moves using opponent_model
              for move in legal_moves:
                  board.push(move)
                  state_tensor = board_to_tensor(board)
                  value = opponent_model(state_tensor).item()
                  move_values.append(value)
                  board.pop()

              # Select the move with the highest predicted value
              # (Opponent plays greedily for simplicity)
              max_value = max(move_values)
              best_moves = [move for i, move in enumerate(legal_moves) 
                            if move_values[i] == max_value]
              chosen_move = np.random.choice(best_moves)
            else:
                chosen_move = np.random.choice(legal_moves)

            # Make the chosen move
            board.push(chosen_move)
            # Store the state after opponent's move
            states.append(board_to_tensor(board))
    
    # Game has ended; determine the reward from AI's perspective
    result = board.result()
    win_reward = reward_decay ** turn_count # favors shorter games
    loss_reward = -reward_decay # a loss is a loss
    if result == '1-0':
        reward = win_reward if ai_plays_white else loss_reward
    elif result == '0-1':
        reward = loss_reward if ai_plays_white else win_reward
    else:
        # Stalemate
        # The midpoint between the win and the loss
        reward = (win_reward + loss_reward) / 2  

    # Add the episode reward to the total reward
    total_reward += reward

    # Assign the reward to all states (from AI's perspective)
    rewards = [reward] * len(states)

    # Update the network using mean squared error loss
    optimizer.zero_grad()
    state_tensors = torch.cat(states, dim=0)
    state_values = trainee_model(state_tensors).squeeze()
    targets = torch.tensor(rewards, device=device)
    loss = nn.functional.mse_loss(state_values, targets)
    loss.backward()
    optimizer.step()

    # Decay the temperature
    temperature = max(min_temperature, temperature * temperature_decay)

    # Calculate mean reward up to the current episode
    mean_reward = total_reward / (episode + 1)

    # Print progress every 10 episodes
    if (episode + 1) % 10 == 0:
        print(f'Episode {episode + 1}/{num_episodes}, '
              + f'Mean Reward: {mean_reward:.4f}, '
              + f'Reward: {reward}, '
              + f'Turns: {turn_count}, '
              + f'Temperature: {temperature:.4f}')

# Save the model after all episodes are completed
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_filename = f"RedChessAI{timestamp}.pth"
model_path = os.path.join(models_dir, model_filename)
torch.save(trainee_model.state_dict(), model_path)
print(f"Model saved to {model_path}")
