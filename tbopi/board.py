import numpy as np

def get_free_cols(board: np.ndarray) -> list:
    return [c for c in range(board.shape[1]) if board[0, c] == 0]


def apply_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    new_board = board.copy()
    for r in reversed(range(board.shape[0])):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            break
    return new_board


def check_winner(board: np.ndarray, player: int) -> bool:
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    for r in range(rows - 3):
        for c in range(cols):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    for r in range(rows - 3):
        for c in range(3, cols):
            if all(board[r + i, c - i] == player for i in range(4)):
                return True
    return False
