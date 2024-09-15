import sys
import chess
import random

def main():
    board = chess.Board()
    while True:
        command = sys.stdin.readline().strip()
        if command == 'uci':
            print('id name RandomEngine')
            print('id author SP-14-Red-ChessAI')
            print('uciok')
        elif command == 'isready':
            print('readyok')
        elif command.startswith('ucinewgame'):
            board.reset()
        elif command.startswith('position'):
            parse_position(command, board)
        elif command.startswith('go'):
            move = select_random_move(board)
            print(f'bestmove {move}')
        elif command == 'quit':
            break
        else:
            print("unhandled command")

def parse_position(command, board):
    tokens = command.split()
    if 'startpos' in tokens:
        board.set_fen(chess.STARTING_FEN)
        moves_index = tokens.index('moves') + 1 if 'moves' in tokens else None
    elif 'fen' in tokens:
        fen_index = tokens.index('fen') + 1
        # Adjust for potential missing fields in FEN string
        fen_parts = tokens[fen_index:fen_index+6]
        fen = ' '.join(fen_parts)
        board.set_fen(fen)
        moves_index = tokens.index('moves') + 1 if 'moves' in tokens else None
    else:
        moves_index = None

    if moves_index:
        moves = tokens[moves_index:]
        for move in moves:
            board.push_uci(move)

def select_random_move(board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return '0000'  # No legal moves available
    move = random.choice(legal_moves)
    return move.uci()

if __name__ == '__main__':
    main()
