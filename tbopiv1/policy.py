import numpy as np
from connect4.policy import Policy

class TBOPIPolicy(Policy):

    def __init__(self, n_trials: int = 200, seed: int = 42):
        self.n_trials = n_trials
        self.seed = seed
        self._rng = None

    def mount(self, _timeout: float | None = None) -> None:
        self._rng = np.random.RandomState(self.seed)

    def act(self, board: np.ndarray) -> int:
        if self._rng is None:
            self.mount()
        rng = self._rng
        player = -1 if np.sum(board == -1) == np.sum(board == 1) else 1
        opp = -player
        legal = get_free_cols(board)

        # Ganar o bloquear de inmediato si es posible
        for col in legal:
            if check_winner(apply_move(board, col, player), player):
                return col
        for col in legal:
            if check_winner(apply_move(board, col, opp), opp):
                return col

        # TBOPI: trials aleatorios + PImp
        q_local = {a: 0.0 for a in legal}
        n_local = {a: 0   for a in legal}

        for _ in range(self.n_trials):
            a = int(rng.choice(legal))
            reward = run_trial(apply_move(board, a, player), player, rng)
            n_local[a] += 1
            q_local[a] += (reward - q_local[a]) / n_local[a]

        action, _ = sample_action_from_inner_stats(
            q_local=q_local,
            n_local=n_local,
            legal_actions=legal,
            rng=rng,
        )
        return action
    
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

def sample_action_from_inner_stats(
    q_local: dict[int, float],
    n_local: dict[int, int],
    legal_actions: list[int],
    *,
    use_q: bool = True,
    use_counts: bool = True,
    rng: np.random.RandomState,
    eps: float = 1e-6,
) -> tuple[int, dict[int, float]]:
    """
    Paso PImp de TBOPI.

    Garantías:
    - p(a) > 0 para toda acción legal (por ε).
    - Valores negativos no penalizan (clip a 0).
    - Acciones más visitadas tienen más peso (confianza).
    """
    actions = list(legal_actions)
    if not actions:
        raise ValueError("legal_actions está vacío.")

    scores = []
    for a in actions:
        q_val  = q_local.get(a, 0.0) if use_q     else 0.0
        n_val  = n_local.get(a, 0)   if use_counts else 0
        scores.append((max(q_val, 0.0) + eps) * (n_val + 1))

    total      = sum(scores)
    probs_list = [s / total for s in scores]
    probs      = {a: p for a, p in zip(actions, probs_list)}
    chosen     = rng.choice(actions, p=probs_list)

    return int(chosen), probs