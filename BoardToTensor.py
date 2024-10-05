import torch
import numpy as np
import chess

# Function to convert the chess board state to a tensor representation
def board_to_tensor(board: chess.Board):
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
    return torch.from_numpy(board_tensor)