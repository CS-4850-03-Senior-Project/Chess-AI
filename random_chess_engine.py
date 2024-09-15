import sys
import chess
import random

def main():
    # Initialize the chess board to the starting position
    board = chess.Board()
    while True:
        # Read a command from the standard input (UCI protocol)
        command = sys.stdin.readline().strip()
        if command == 'uci':
            # Engine identification response
            print('id name RandomEngine')
            print('id author SP-14-Red-ChessAI')
            print('uciok')
        elif command == 'isready':
            # Signal that the engine is ready
            print('readyok')
        elif command.startswith('ucinewgame'):
            # Reset the board for a new game
            board.reset()
        elif command.startswith('position'):
            # Set up the board position as per the command
            parse_position(command, board)
        elif command.startswith('go'):
            # Generate and output a move
            move = select_random_move(board)
            print(f'bestmove {move}')
        elif command == 'quit':
            # Exit the engine loop
            break
        else:
            # Handle any other unrecognized commands
            print("unhandled command")

def parse_position(command, board):
    # Split the command into tokens for parsing
    tokens = command.split()
    if 'startpos' in tokens:
        # Set up the board to the standard starting position
        board.set_fen(chess.STARTING_FEN)
        # Find the index of the 'moves' keyword if it exists
        moves_index = tokens.index('moves') + 1 if 'moves' in tokens else None
    elif 'fen' in tokens:
        # Find the index where the FEN string starts
        fen_index = tokens.index('fen') + 1
        # Extract the FEN string components (6 fields)
        fen_parts = tokens[fen_index:fen_index+6]
        fen = ' '.join(fen_parts)
        # Set up the board with the given FEN position
        board.set_fen(fen)
        # Find the index of the 'moves' keyword if it exists
        moves_index = tokens.index('moves') + 1 if 'moves' in tokens else None
    else:
        # No position specified
        moves_index = None

    # If moves are specified after the position, apply them
    if moves_index:
        moves = tokens[moves_index:]
        for move in moves:
            # Apply each move to the board using UCI notation
            board.push_uci(move)

def select_random_move(board):
    # Get a list of all legal moves from the current position
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # If no legal moves are available, indicate game over (e.g., checkmate or stalemate)
        return '0000'  # '0000' is sometimes used in UCI to indicate no move
    # Randomly select one of the legal moves
    move = random.choice(legal_moves)
    # Return the move in UCI format (e.g., 'e2e4')
    return move.uci()

if __name__ == '__main__':
    main()
