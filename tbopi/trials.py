import numpy as np
from groups.tbopi.board import get_free_cols, apply_move, check_winner


def run_trial(board: np.ndarray, player: int, rng: np.random.RandomState) -> float:
    """Rollout aleatorio puro hasta estado terminal."""
    b, current = board.copy(), player
    while True:
        free = get_free_cols(b)
        if not free:
            return 0.0
        b = apply_move(b, int(rng.choice(free)), current)
        if check_winner(b, current):
            return 1.0 if current == player else -1.0
        current = -current
