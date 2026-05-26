"""
TBOPI (Trial-Based Online Policy Improvement) para Connect 4.



Mejoras aplicadas respecto a la versión original:
  1. Selección de trial con UCB en lugar de muestreo uniforme
  2. Rollout ε-greedy con heurística de columnas centrales
"""

from __future__ import annotations
import math
import numpy as np
from connect4.policy import Policy

# ---------------------------------------------------------------------------
# Lógica de juego (sin estado, funciones puras)
# ---------------------------------------------------------------------------

def get_free_cols(board: np.ndarray) -> list[int]:
    """Columnas con al menos una celda libre."""
    return [c for c in range(board.shape[1]) if board[0, c] == 0]

def apply_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """Devuelve una copia del tablero con el movimiento aplicado."""
    new_board = board.copy()
    for row in reversed(range(board.shape[0])):
        if new_board[row, col] == 0:
            new_board[row, col] = player
            break
    return new_board

def check_winner(board: np.ndarray, player: int) -> bool:
    """True si `player` tiene cuatro en línea."""
    # Horizontal
    for r in range(6):
        for c in range(4):
            if board[r, c] == player and board[r, c+1] == player and board[r, c+2] == player and board[r, c+3] == player:
                return True
    # Vertical
    for c in range(7):
        for r in range(3):
            if board[r, c] == player and board[r+1, c] == player and board[r+2, c] == player and board[r+3, c] == player:
                return True
    # Diagonales
    for r in range(3):
        for c in range(4):
            if board[r, c] == player and board[r+1, c+1] == player and board[r+2, c+2] == player and board[r+3, c+3] == player:
                return True
        for c in range(3, 7):
            if board[r, c] == player and board[r+1, c-1] == player and board[r+2, c-2] == player and board[r+3, c-3] == player:
                return True
    return False

def current_player(board: np.ndarray) -> int:
    """
    Infiere qué jugador le toca a partir del conteo de fichas.
    Convención: +1 mueve primero.
    """
    return -1 if np.sum(board == -1) == np.sum(board == 1) else 1


# ---------------------------------------------------------------------------
# Heurística de rollout (mejora 2)
# ---------------------------------------------------------------------------

def rollout_policy_eps_greedy(
    board: np.ndarray,
    player: int,
    rng: np.random.RandomState,
    eps: float = 0.3,
    last_move: int | None = None,
    lgr_table: dict[int, dict[int, int]] | None = None,
) -> int:
    """
    Política de rollout ε-greedy con heurística de columna central.
    Añade LGRF: Prioriza la última respuesta exitosa registrada.
    """
    free = get_free_cols(board)
    if not free:
        raise ValueError("No hay columnas libres.")
    
    if rng.random() < eps:
        return int(rng.choice(free))

    # 1. LGRF: Intentar la respuesta que funcionó antes para el movimiento del rival
    if lgr_table is not None and last_move is not None:
        suggested = lgr_table[player].get(last_move)
        if suggested is not None and board[0, suggested] == 0:
            return suggested

    # 2. Selección central rápida
    for c in [3, 2, 4, 1, 5, 0, 6]:
        if board[0, c] == 0:
            return c
    return int(rng.choice(free))


# ---------------------------------------------------------------------------
# Motor de simulaciones
# ---------------------------------------------------------------------------

def run_trial(
    board: np.ndarray,
    root_player: int,
    rng: np.random.RandomState,
    last_move: int,
    lgr_table: dict[int, dict[int, int]],
    rollout_eps: float = 0.3,
) -> float:
    """
    Rollout hasta estado terminal con política ε-greedy.
    Actualiza la tabla LGR si hay un ganador.
    """
    b = board.copy()
    current = -root_player  # El oponente mueve después del movimiento inicial 'a'
    trial_history = []
    
    while True:
        if not get_free_cols(b):
            return 0.0
        
        col = rollout_policy_eps_greedy(
            b, current, rng, eps=rollout_eps, last_move=last_move, lgr_table=lgr_table
        )
        trial_history.append((current, last_move, col))
        
        for row in reversed(range(6)):
            if b[row, col] == 0:
                b[row, col] = current
                break
                
        if check_winner(b, current):
            winner = current
            # Refuerzo LGRF: Guardar respuestas del ganador
            for p, prev_m, resp in trial_history:
                if p == winner:
                    lgr_table[p][prev_m] = resp
            return 1.0 if winner == root_player else -1.0
        
        last_move = col
        current = -current

# ---------------------------------------------------------------------------
# Selección de acción con UCB (mejora 1)
# ---------------------------------------------------------------------------

def _ucb_select(
    q_local: dict[int, float],
    n_local: dict[int, int],
    legal: list[int],
    total_n: int,
    c: float = math.sqrt(2),
) -> int:
    """
    Elige la acción con mayor valor UCB.

    UCB(a) = Q(a) + c * sqrt(ln(N) / n(a))

    Las acciones no visitadas reciben prioridad máxima.
    """
    # Priorizar no visitadas con orden central
    for a in [3, 2, 4, 1, 5, 0, 6]:
        if a in legal and n_local.get(a, 0) == 0:
            return a

    best_action, best_score = legal[0], -math.inf
    log_n = math.log(total_n) if total_n > 0 else 0.0
    for a in legal:
        n_a = n_local.get(a, 0)
        score = q_local.get(a, 0.0) + c * math.sqrt(log_n / n_a)
        if score > best_score:
            best_score, best_action = score, a
    return best_action


# ---------------------------------------------------------------------------
# Política principal
# ---------------------------------------------------------------------------


class TBOPIPolicy(Policy):
    """
    Trial-Based Online Policy Improvement para Connect 4.

    Parámetros
    ----------
    n_trials:
        Número de simulaciones Monte Carlo por turno.
    seed:
        Semilla para reproducibilidad.
    ucb_c:
        Constante de exploración UCB (sqrt(2) es el estándar teórico).
    rollout_eps:
        Probabilidad de jugada aleatoria en el rollout (0 = greedy puro).
    """

    def __init__(
        self,
        n_trials: int = 200,
        seed: int = 42,
        ucb_c: float = math.sqrt(2),
        rollout_eps: float = 0.3,
    ) -> None:
        self.n_trials = n_trials
        self.seed = seed
        self.ucb_c = ucb_c
        self.rollout_eps = rollout_eps
        self._rng: np.random.RandomState | None = None

    def mount(self, _timeout: float | None = None) -> None:
        self._rng = np.random.RandomState(self.seed)

    def act(self, board: np.ndarray) -> int:
        if self._rng is None:
            self.mount()
        rng = self._rng

        player = current_player(board)
        opp = -player
        legal = get_free_cols(board)

        # --- Respuesta inmediata: ganar o bloquear ---
        for col in legal:
            if check_winner(apply_move(board, col, player), player):
                return col
        for col in legal:
            if check_winner(apply_move(board, col, opp), opp):
                return col

        # --- Fase TBOPI: trials con selección UCB ---
        q_local: dict[int, float] = {a: 0.0 for a in legal}
        n_local: dict[int, int] = {a: 0 for a in legal}
        lgr_table: dict[int, dict[int, int]] = {1: {}, -1: {}}
        total_n = 0

        for _ in range(self.n_trials):
            a = _ucb_select(q_local, n_local, legal, total_n, c=self.ucb_c)
            reward = run_trial(
                apply_move(board, a, player),
                player,
                rng,
                last_move=a,
                lgr_table=lgr_table,
                rollout_eps=self.rollout_eps,
            )
            total_n += 1
            n_local[a] += 1
            # Media incremental
            q_local[a] += (reward - q_local[a]) / n_local[a]

        # --- Paso PImp: selección greedy sobre Q local ---
        best_q = max(q_local.values())
        best_actions = [a for a in legal if q_local[a] == best_q]
        return int(rng.choice(best_actions))