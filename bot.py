import chess
import torch
from BoardToTensor import board_to_tensor
from EvaluationNetwork import EvaluationNetwork
import os

# class Evaluation:
#   def __init__(self, move, score):
#     self.move = move
#     self.score = score

class RedChessAIBot:
  def __init__(self, model_path, device):
    checkpoint = torch.load(model_path)
    model = EvaluationNetwork().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    self.device = device
    self.model = model

  def evaluate_position(self, board: chess.Board):
    tensor = board_to_tensor(board).to(self.device)
    return float(self.model(tensor).item())

  def evaluate_next(self, board: chess.Board):
    legal_moves = list(board.legal_moves)
    scores = {}
    for move in legal_moves:
      board.push(move)
      evaluation = self.evaluate_position(board)
      scores[str(move)] = evaluation
      board.pop()
    return scores

if __name__ == '__main__':
  # Device configuration (I have multiple GPUs; I want to use device 0)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  models_dir = "MODELS"
  model_filename = "Pretrained.pth" 
  model_path = os.path.join(models_dir, model_filename)
  bot = RedChessAIBot(model_path, device)

  """
  6R1/8/3kp3/3b1pP1/2B5/1P2P1r1/3K4/8 b - - 3 45,+13
  6R1/8/3kp3/5pP1/2b5/1P2P1r1/3K4/8 w - - 0 46,+123
  6R1/8/3kp3/5pP1/2P5/4P1r1/3K4/8 b - - 0 46,+105
  6R1/8/4p3/2k2pP1/2P5/4P1r1/3K4/8 w - - 1 47,+350
  6R1/8/4p1P1/2k2p2/2P5/4P1r1/3K4/8 b - - 0 47,+373
  6R1/8/3kp1P1/5p2/2P5/4P1r1/3K4/8 w - - 1 48,+888
  6R1/8/3kp1P1/2P2p2/8/4P1r1/3K4/8 b - - 0 48,+1863
  6R1/2k5/4p1P1/2P2p2/8/4P1r1/3K4/8 w - - 1 49,+1276
  6R1/2k3P1/4p3/2P2p2/8/4P1r1/3K4/8 b - - 0 49,+3113
  6R1/1k4P1/4p3/2P2p2/8/4P1r1/3K4/8 w - - 1 50,+1088
  6R1/1k4P1/2P1p3/5p2/8/4P1r1/3K4/8 b - - 0 50,+4796
  """

  fen = "6R1/8/3kp3/3b1pP1/2B5/1P2P1r1/3K4/8 b - - 3 45"
  board = chess.Board(fen)
  next = bot.evaluate_next(board)
  for move in next:
    print(f'{move} {next[move]:.3f}')
