import chess
from BoardToTensor import board_to_tensor
import torch
import os
from EvaluationNetworkTanh import EvaluationNetwork

# Some data to test
"""FEN,Evaluation
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1,-10
rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2,+56
rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2,-9
rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3,+52
rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPPN1PPP/R1BQKBNR b KQkq - 1 3,-26
rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/8/PPPN1PPP/R1BQKBNR w KQkq - 2 4,+50
rnbqkb1r/ppp2ppp/4pn2/3pP3/3P4/8/PPPN1PPP/R1BQKBNR b KQkq - 0 4,+10
rnbqkb1r/pppn1ppp/4p3/3pP3/3P4/8/PPPN1PPP/R1BQKBNR w KQkq - 1 5,+75
rnbqkb1r/pppn1ppp/4p3/3pP3/3P1P2/8/PPPN2PP/R1BQKBNR b KQkq - 0 5,+52
rnbqkb1r/pp1n1ppp/4p3/2ppP3/3P1P2/8/PPPN2PP/R1BQKBNR w KQkq - 0 6,+52
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
6R1/1k4P1/2P1p3/5p2/8/4P1r1/3K4/8 b - - 0 50,+4796"""

# Device configuration (I have multiple GPUs; I want to use device 0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Getting ahead of myself, let's load the model
model_dir = "MODELS"
model_filename = "Pretrained20241008172417.pth"
model_path = os.path.join(model_dir,model_filename)

checkpoint = torch.load(model_path)

model = EvaluationNetwork().to(device)
model.load_state_dict(checkpoint['model_state_dict'])

while True:
  fen_input = input("Enter a FEN string (or type 'exit' to quit): ")

  if(fen_input.lower() == 'exit'):
    break

  try:
    board = chess.Board(fen_input)
    board_tensor = board_to_tensor(board).to(device)
    with torch.no_grad():
      value = model(board_tensor).item()
    print(value)
  except:
    print(f'Invalid input!')
