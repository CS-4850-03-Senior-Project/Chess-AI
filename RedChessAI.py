# Title: Red Chess AI
# Authors:  James-Calvin Meaders,
#           Nhut Tran,  
#           David Cardenas-Verdin,
#           Joe Nguyen
# Date: 2024 Fall Semester November 21
# Description:  An AI agent for playing chess. The model is intended to be 
#   available for play at https://lichess.org/@/RedChessAI
# Things to add or learn about:
#   - Temporal difference (TD) Learning
#   - Saving optimizer state
#   - Start training at endgame
# Development History:
#   - Started with an arbitrary model size and simple reward function
#   - Added saving and load models
#   - Model size increased via hidden layers `intermediate_layers`
#   - Reward function modified to incentivize shorter games
#   - Added reward for capturing pieces
# References:
#   - How to Create a Chess Engine with PyTorch
#       - https://www.youtube.com/watch?v=-WDRaRUjRMg
#   - How to Evaluate a Chess Position
#       - https://thechessworld.com/articles/general-information/how-to-evaluate-a-chess-position/
#   - How to Evaluate Chess Positions (Example)
#       - https://www.chess.com/article/view/how-to-evaluate-a-position
#   - Q-Learning
#       - https://www.geeksforgeeks.org/q-learning-in-python/
#   - Reinforcement Learning in Chess
#       - https://medium.com/@samgill1256/reinforcement-learning-in-chess-73d97fad96b3
# Things to test:
#   - board_to_tensor()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import chess.engine
import os
from datetime import datetime
from EvaluationNetwork import EvaluationNetwork
from BoardToTensor import board_to_tensor


# Hyperparameters
learning_rate = 0.01
num_episodes = 1000
initial_temperature = 0.5
temperature_decay = 0.995
min_temperature = 0.05
opponent_temperature = 0.25

# Reward Parameters
reward_decay = 0.99
win_turn_bonus_decay = 0.95
target_turn_count = 80
win_reward = 1.0 # reward for winning
win_turn_bonus = 1.0 # bonus points for finishing the game within target turns
capture_rewards = {
    chess.PAWN: 0.01,
    chess.KNIGHT: 0.025,
    chess.BISHOP: 0.025,
    chess.ROOK: 0.05,
    chess.QUEEN: 0.10
}
stalemate_reward = 0.0
stalemate_turn_bonus = 1.0
stalemate_turn_decay = 0.9
stalemate_target_turn = 40

# Starting point of trainee model
load_trainee = True
trainee_model_filename = "Pretrained.pth" 

# Model of the opponent (can be the same as trainee)
load_opponent = True # If false, the opponent will play randomly
opponent_model_load_file = "Pretrained.pth"

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

# # source: https://stackoverflow.com/questions/58556338/python-evaluating-a-board-position-using-stockfish-from-the-python-chess-librar
# def stockfish_evaluation(board, time_limit = 0.01):
#     engine = chess.engine.SimpleEngine.popen_uci("engines\stockfish\stockfish-windows-x86-64-avx2.exe")
#     result = engine.analyse(board, chess.engine.Limit(time=time_limit))
#     score = result['score'].relative.score()
#     if score == None:
#         return -1000
#     else:
#         return score

# Initialize the value network and optimizer
trainee_model = EvaluationNetwork().to(device)
optimizer = optim.Adam(trainee_model.parameters(), lr=learning_rate)
opponent_model = EvaluationNetwork().to(device)

# Load trainee
if load_trainee:
  model_path = os.path.join(models_dir, trainee_model_filename)
  trainee_checkpoint = torch.load(model_path)
  trainee_model.load_state_dict(trainee_checkpoint['model_state_dict'])
  optimizer.load_state_dict(trainee_checkpoint['optimizer_state_dict'])
trainee_model.to(device) # This may not be necessary

# Load opponent
if load_opponent:
  opponent_model_path = os.path.join(models_dir, opponent_model_load_file)
  opponent_checkpoint = torch.load(opponent_model_path)
  opponent_model.load_state_dict(opponent_checkpoint['model_state_dict']) 

def save_model():
    # Save the model after all episodes are completed
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_filename = f"RedChessAI{timestamp}.pth"
    model_path = os.path.join(models_dir, model_filename)
    torch.save({
        'model_state_dict': trainee_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")

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
                state_tensor = board_to_tensor(board).unsqueeze(0).to(device)
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

            # Calculating capture rewards
            reward = 0.0
            if board.is_capture(chosen_move):
                captured_piece = board.piece_at(chosen_move.to_square)
                if captured_piece is not None:
                    if captured_piece.color != board.turn:
                        captured_piece_type = captured_piece.piece_type
                        reward += capture_rewards.get(captured_piece_type, 0.0)

            # Make the chosen move
            board.push(chosen_move)
            # Store the state after the move
            states.append(board_to_tensor(board).unsqueeze(0).to(device))
            # Add capture reward
            rewards.append(reward)

        else:
            # Opponent's turn
            legal_moves = list(board.legal_moves)

            if load_opponent:
                move_values = []
                # Evaluate all legal moves using opponent_model
                for move in legal_moves:
                        board.push(move)
                        state_tensor = board_to_tensor(board).unsqueeze(0).to(device)
                        value = opponent_model(state_tensor).item()
                        move_values.append(value)
                        board.pop()

                # Apply softmax with temperature to the predicted values
                logits = np.array(move_values) / opponent_temperature
                probabilities = np.exp(logits - np.max(logits))
                probabilities /= probabilities.sum()
                
                move_index = np.random.choice(len(legal_moves), p=probabilities)
                chosen_move = legal_moves[move_index]
            else:
                chosen_move = np.random.choice(legal_moves)

            # Make the chosen move
            board.push(chosen_move)
    
    # Game has ended; determine the reward from AI's perspective
    offset_turns = max(turn_count - target_turn_count, 0)
    win_bonus = win_turn_bonus * win_turn_bonus_decay ** offset_turns
    match_win_reward = win_reward + win_bonus
    loss_reward = -1
    result = board.result()
    if result == '1-0':
        final_reward = match_win_reward if ai_plays_white else loss_reward
    elif result == '0-1':
        final_reward = loss_reward if ai_plays_white else match_win_reward
    else:
        # Stalemate
        stalemate_turn_offset = max(turn_count - stalemate_target_turn, 0)
        stalemate_percentage = stalemate_turn_decay ** stalemate_turn_offset
        stalemate_actual_bonus = stalemate_turn_bonus * stalemate_percentage
        final_reward = stalemate_reward + stalemate_actual_bonus

    # Add the final game reward to the last move's reward
    if rewards:
        rewards[-1] += final_reward
    else:
        rewards.append(final_reward)

    # Compute discounted cumulative rewards
    def compute_returns(rewards, gamma):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    
    returns = compute_returns(rewards, reward_decay)

    # Assign the reward to all states (from AI's perspective)
    # rewards = [reward] * len(states)

    # Update the network using mean squared error loss
    optimizer.zero_grad()
    state_tensors = torch.cat(states, dim=0)
    state_values = trainee_model(state_tensors).squeeze()
    targets = torch.tensor(returns, device=device)
    loss = nn.functional.mse_loss(state_values, targets)
    loss.backward()
    optimizer.step()

    # Decay the temperature
    temperature = max(min_temperature, temperature * temperature_decay)

    # Calculate mean reward up to the current episode
    total_reward += sum(rewards)
    mean_reward = total_reward / (episode + 1)

    # Print progress every 10 episodes
    if (episode + 1) % 10 == 0:
        print(f'Episode {episode + 1}/{num_episodes}, '
              + f'Mean Reward: {mean_reward:.3f}, '
              + f'Episode Reward: {sum(rewards):.3f}, '
              + f'Turns: {turn_count}, '
              + f'Temperature: {temperature:.3f}')
        if(episode + 1) % 100 == 0:
            save_model()

save_model()