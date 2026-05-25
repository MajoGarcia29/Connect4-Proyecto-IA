"""
TBOPI (Trial-Based Online Policy Improvement) para Connect 4.

Mejoras aplicadas respecto a la versión original:
  1. Selección de trial con UCB en lugar de muestreo uniforme
     (Kocsis & Szepesvári, 2006; Coulom, 2006)
  2. Rollout ε-greedy con heurística de columnas centrales
     (Silver et al., 2016; Gelly & Silver, 2007)
  3. Softmax con temperatura en PImp en lugar de pesos lineales
     (Rosin, 2011; Browne et al., 2012)

Pendiente de implementar:
  4. Tree reuse entre turnos (Chiara et al., 2019)
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
    rows, cols = board.shape
    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    # Diagonal ascendente
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    # Diagonal descendente
    for r in range(rows - 3):
        for c in range(3, cols):
            if all(board[r + i, c - i] == player for i in range(4)):
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

def _col_center_weight(col: int, n_cols: int) -> float:
    """Preferencia por columnas centrales (valor en [0, 1])."""
    center = (n_cols - 1) / 2.0
    return 1.0 - abs(col - center) / center


def rollout_policy_eps_greedy(
    board: np.ndarray,
    player: int,
    rng: np.random.RandomState,
    eps: float = 0.3,
) -> int:
    """
    Política de rollout ε-greedy con heurística de columna central.

    Con probabilidad eps juega aleatoriamente; de lo contrario elige
    la columna libre más cercana al centro (Silver et al., 2016).
    """
    free = get_free_cols(board)
    if not free:
        raise ValueError("No hay columnas libres.")
    if rng.random() < eps:
        return int(rng.choice(free))
    weights = np.array([_col_center_weight(c, board.shape[1]) for c in free])
    weights /= weights.sum()
    return int(rng.choice(free, p=weights))


# ---------------------------------------------------------------------------
# Motor de simulaciones
# ---------------------------------------------------------------------------

def run_trial(
    board: np.ndarray,
    player: int,
    rng: np.random.RandomState,
    rollout_eps: float = 0.3,
) -> float:
    """
    Rollout hasta estado terminal con política ε-greedy.

    Retorna +1 si gana `player`, -1 si pierde, 0 si empate.
    """
    b, current = board.copy(), player
    while True:
        free = get_free_cols(b)
        if not free:
            return 0.0
        col = rollout_policy_eps_greedy(b, current, rng, eps=rollout_eps)
        b = apply_move(b, col, current)
        if check_winner(b, current):
            return 1.0 if current == player else -1.0
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

    Las acciones no visitadas reciben prioridad máxima (inf).
    Referencia: Kocsis & Szepesvári (2006), Coulom (2006).
    """
    best_action, best_score = legal[0], -math.inf
    log_n = math.log(total_n) if total_n > 0 else 0.0
    for a in legal:
        n_a = n_local.get(a, 0)
        if n_a == 0:
            return a  # acción no visitada: explorar primero
        exploration = c * math.sqrt(log_n / n_a)
        score = q_local.get(a, 0.0) + exploration
        if score > best_score:
            best_score, best_action = score, a
    return best_action


# ---------------------------------------------------------------------------
# Paso PImp con softmax (mejora 3)
# ---------------------------------------------------------------------------

def sample_action_from_inner_stats(
    q_local: dict[int, float],
    n_local: dict[int, int],
    legal_actions: list[int],
    *,
    rng: np.random.RandomState,
    temperature: float = 1.0,
    use_q: bool = True,
    use_counts: bool = True,
    eps: float = 1e-6,
) -> tuple[int, dict[int, float]]:
    """
    Paso PImp de TBOPI con softmax parametrizado por temperatura.

    score(a) = (max(Q(a), 0) + eps) * (n(a) + 1)
    p(a) ∝ exp(score(a) / temperature)

    Con temperature → 0 se vuelve greedy; con temperature → ∞, uniforme.
    Garantiza p(a) > 0 para toda acción legal.

    Referencias: Rosin (2011), Browne et al. (2012).
    """
    actions = list(legal_actions)
    if not actions:
        raise ValueError("legal_actions está vacío.")

    raw_scores = []
    for a in actions:
        q_val = q_local.get(a, 0.0) if use_q else 0.0
        n_val = n_local.get(a, 0) if use_counts else 0
        raw_scores.append((max(q_val, 0.0) + eps) * (n_val + 1))

    # Softmax con temperatura (estable numéricamente)
    scaled = np.array(raw_scores) / temperature
    scaled -= scaled.max()  # estabilidad numérica
    exp_scores = np.exp(scaled)
    probs_array = exp_scores / exp_scores.sum()

    probs = dict(zip(actions, probs_array.tolist()))
    chosen = int(rng.choice(actions, p=probs_array))
    return chosen, probs


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
    temperature:
        Temperatura del softmax en PImp (1.0 = sin sesgo adicional).
    """

    def __init__(
        self,
        n_trials: int = 200,
        seed: int = 42,
        ucb_c: float = math.sqrt(2),
        rollout_eps: float = 0.3,
        temperature: float = 1.0,
    ) -> None:
        self.n_trials = n_trials
        self.seed = seed
        self.ucb_c = ucb_c
        self.rollout_eps = rollout_eps
        self.temperature = temperature
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
        total_n = 0

        for _ in range(self.n_trials):
            a = _ucb_select(q_local, n_local, legal, total_n, c=self.ucb_c)
            reward = run_trial(
                apply_move(board, a, player),
                player,
                rng,
                rollout_eps=self.rollout_eps,
            )
            total_n += 1
            n_local[a] += 1
            # Media incremental (Welford)
            q_local[a] += (reward - q_local[a]) / n_local[a]

        # --- Paso PImp con softmax ---
        action, _ = sample_action_from_inner_stats(
            q_local=q_local,
            n_local=n_local,
            legal_actions=legal,
            rng=rng,
            temperature=self.temperature,
        )
        return action