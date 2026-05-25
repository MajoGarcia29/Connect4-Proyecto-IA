"""
TBOPI (Trial-Based Online Policy Improvement) para Connect 4.

Flujo por turno:
  1. Respuesta inmediata: ganar o bloquear si es posible.
  2. Trials con UCB + LGRF: cada trial elige acción con UCB, hace rollout
     con política LGRF → Q-table → ε-greedy (en ese orden de prioridad),
     y actualiza tanto Q(a) local como la Q-table persistente.
  3. PImp: softmax sobre Q local para samplear la acción final.

Conocimiento persistente (Q-table por features):
  La Q-table asocia (player, feature_vector, col) → Q-value y se guarda
  en disco entre partidas. Las features capturan el contexto estratégico
  del tablero: amenazas propias, amenazas del oponente y control del centro.
  Esto permite generalizar entre posiciones similares nunca vistas exactamente.

Referencias:
  - Kocsis & Szepesvári (2006): UCT
  - Coulom (2006): MCTS con UCB
  - Rosin (2011): softmax/temperatura en selección final
  - Baier & Winands (2014): LGRF (Last Good Reply with Forgetting)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

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


def current_player(board: np.ndarray) -> int:
    """
    Infiere qué jugador le toca a partir del conteo de fichas.
    Convención: +1 mueve primero.
    """
    return -1 if np.sum(board == -1) == np.sum(board == 1) else 1


# ---------------------------------------------------------------------------
# Features del tablero para la Q-table persistente
# ---------------------------------------------------------------------------

def _count_open_threats(board: np.ndarray, player: int, length: int) -> int:
    """
    Cuenta secuencias de `length` fichas de `player` con al menos
    un extremo libre. Estas son amenazas reales que pueden completarse.
    """
    rows, cols = board.shape
    count = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(rows):
        for c in range(cols):
            for dr, dc in directions:
                end_r = r + dr * (length - 1)
                end_c = c + dc * (length - 1)
                if not (0 <= end_r < rows and 0 <= end_c < cols):
                    continue
                cells = [board[r + dr * i, c + dc * i] for i in range(length)]
                if all(v == player for v in cells):
                    # Verificar al menos un extremo libre
                    before_r, before_c = r - dr, c - dc
                    after_r,  after_c  = r + dr * length, c + dc * length
                    before_free = (0 <= before_r < rows and 0 <= before_c < cols
                                   and board[before_r, before_c] == 0)
                    after_free  = (0 <= after_r  < rows and 0 <= after_c  < cols
                                   and board[after_r,  after_c]  == 0)
                    if before_free or after_free:
                        count += 1
    return count


def board_features(board: np.ndarray, player: int) -> tuple[int, int, int]:
    """
    Extrae tres features estratégicas del tablero:

      - own_threats:  amenazas propias abiertas de longitud 3
      - opp_threats:  amenazas del oponente abiertas de longitud 3
      - center_ctrl:  fichas propias en las 3 columnas centrales

    El vector resultante tiene un espacio pequeño (~cientos de combinaciones)
    y generaliza entre posiciones con la misma estructura estratégica.
    """
    opp         = -player
    cols        = board.shape[1]
    center_cols = range(cols // 2 - 1, cols // 2 + 2)

    own_threats = _count_open_threats(board, player, length=3)
    opp_threats = _count_open_threats(board, opp,    length=3)
    center_ctrl = int(np.sum(board[:, list(center_cols)] == player))

    return (own_threats, opp_threats, center_ctrl)


# ---------------------------------------------------------------------------
# Q-table persistente (aprende entre partidas)
# ---------------------------------------------------------------------------

# Clave: (player, own_threats, opp_threats, center_ctrl, col)
# Valor: Q-value estimado para jugar `col` en ese contexto de features
QKey   = tuple[int, int, int, int, int]
QTable = dict[QKey, float]


def _q_key(player: int, features: tuple[int, int, int], col: int) -> QKey:
    """Construye la clave de la Q-table a partir de contexto y acción."""
    return (player, *features, col)


def load_qtable(path: Path) -> QTable:
    """Carga la Q-table desde disco. Devuelve tabla vacía si no existe."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    # JSON no soporta tuplas como claves; se serializaron como strings
    return {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}


def save_qtable(table: QTable, path: Path) -> None:
    """Guarda la Q-table en disco como JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {",".join(str(x) for x in k): v for k, v in table.items()}
    path.write_text(json.dumps(raw))


def _update_qtable(
    qtable: QTable,
    player: int,
    features: tuple[int, int, int],
    col: int,
    reward: float,
    alpha: float = 0.1,
) -> None:
    """
    Actualiza el Q-value de (features, col) con la recompensa del trial.

    Usa actualización TD(0) incremental:
        Q(s, a) ← Q(s, a) + α * (r - Q(s, a))

    Con α pequeño (0.1) el valor converge lentamente pero es estable
    ante rewards ruidosas de rollouts individuales.
    """
    key = _q_key(player, features, col)
    current_q = qtable.get(key, 0.0)
    qtable[key] = current_q + alpha * (reward - current_q)


# ---------------------------------------------------------------------------
# Tabla LGRF (Last Good Reply with Forgetting) — aprende dentro del turno
# ---------------------------------------------------------------------------

LGRFTable = dict[tuple[int, int], int]


def _update_lgrf(
    table: LGRFTable,
    history: list[tuple[int, int]],
    winner: int,
) -> None:
    """
    Actualiza la tabla LGRF tras un rollout.

    Registra (player, col_oponente) → col_respuesta para cada par
    consecutivo en la historia ganadora. Olvida entradas de historias
    perdedoras. Empates no modifican la tabla.
    """
    if winner == 0:
        return
    for i in range(len(history) - 1):
        _,        opp_col = history[i]
        my_player, my_col = history[i + 1]
        key = (my_player, opp_col)
        if my_player == winner:
            table[key] = my_col
        else:
            table.pop(key, None)


# ---------------------------------------------------------------------------
# Política de rollout: LGRF → Q-table → ε-greedy
# ---------------------------------------------------------------------------

def rollout_policy(
    board: np.ndarray,
    player: int,
    last_opp_col: int | None,
    lgrf_table: LGRFTable,
    qtable: QTable,
    rng: np.random.RandomState,
    eps: float = 0.3,
) -> int:
    """
    Política de rollout con tres niveles de prioridad decreciente:

      1. LGRF: respuesta aprendida en este turno para la última jugada
         del oponente. Conocimiento más específico y reciente.
      2. Q-table: columna con mayor Q-value aprendido entre partidas
         para el contexto de features actual. Conocimiento generalizado
         y persistente que reemplaza la heurística estática de columna central.
      3. ε-greedy aleatorio: exploración pura cuando los niveles
         anteriores no tienen información o por azar (eps).
    """
    free = get_free_cols(board)

    # Nivel 1: LGRF (conocimiento intra-turno)
    if last_opp_col is not None:
        reply = lgrf_table.get((player, last_opp_col))
        if reply is not None and reply in free:
            return reply

    # Nivel 2: Q-table (conocimiento inter-partidas)
    if rng.random() >= eps:
        features = board_features(board, player)
        q_vals = {c: qtable.get(_q_key(player, features, c), 0.0) for c in free}
        best_q = max(q_vals.values())
        best_cols = [c for c, q in q_vals.items() if q == best_q]
        return int(rng.choice(best_cols))

    # Nivel 3: aleatorio
    return int(rng.choice(free))


# ---------------------------------------------------------------------------
# Motor de simulaciones
# ---------------------------------------------------------------------------

def run_trial(
    board: np.ndarray,
    player: int,
    lgrf_table: LGRFTable,
    qtable: QTable,
    rng: np.random.RandomState,
    rollout_eps: float = 0.3,
) -> float:
    """
    Rollout hasta estado terminal con política LGRF → Q-table → ε-greedy.

    Construye el historial de (jugador, columna) durante la simulación
    y al terminar actualiza la tabla LGRF con el resultado.

    Retorna +1 si gana `player`, -1 si pierde, 0 si empate.
    """
    b            = board.copy()
    current      = player
    history: list[tuple[int, int]] = []
    last_opp_col: int | None = None

    while True:
        free = get_free_cols(b)
        if not free:
            _update_lgrf(lgrf_table, history, winner=0)
            return 0.0

        col = rollout_policy(b, current, last_opp_col, lgrf_table, qtable, rng, rollout_eps)
        history.append((current, col))
        b = apply_move(b, col, current)

        if check_winner(b, current):
            _update_lgrf(lgrf_table, history, winner=current)
            return 1.0 if current == player else -1.0

        last_opp_col = col
        current      = -current


# ---------------------------------------------------------------------------
# Selección de acción con UCB
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
    Referencia: Kocsis & Szepesvári (2006), Coulom (2006).
    """
    log_n = math.log(total_n) if total_n > 0 else 0.0
    best_action, best_score = legal[0], -math.inf
    for a in legal:
        n_a = n_local.get(a, 0)
        if n_a == 0:
            return a
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

    Combina TBOPI + LGRF con una Q-table persistente por features del tablero.
    La Q-table se guarda en disco al final de cada turno y se carga al inicio,
    acumulando conocimiento entre partidas sin crecer de forma descontrolada.

    Parámetros
    ----------
    n_trials:
        Número de simulaciones Monte Carlo por turno.
    seed:
        Semilla para reproducibilidad.
    ucb_c:
        Constante de exploración UCB (sqrt(2) es el estándar teórico).
    rollout_eps:
        Probabilidad de jugada aleatoria en el rollout.
    temperature:
        Temperatura del softmax en PImp.
    qtable_path:
        Ruta al archivo JSON donde se persiste la Q-table entre partidas.
    qtable_alpha:
        Tasa de aprendizaje para actualizar la Q-table (TD(0)).
    """

    def __init__(
        self,
        n_trials: int     = 200,
        seed: int         = 42,
        ucb_c: float      = math.sqrt(2),
        rollout_eps: float = 0.3,
        temperature: float = 1.0,
        qtable_path: str  = "qtable.json",
        qtable_alpha: float = 0.1,
    ) -> None:
        self.n_trials     = n_trials
        self.seed         = seed
        self.ucb_c        = ucb_c
        self.rollout_eps  = rollout_eps
        self.temperature  = temperature
        self.qtable_path  = Path(qtable_path)
        self.qtable_alpha = qtable_alpha
        self._rng: np.random.RandomState | None = None
        self._qtable: QTable = {}

    def mount(self, _timeout: float | None = None) -> None:
        self._rng    = np.random.RandomState(self.seed)
        self._qtable = load_qtable(self.qtable_path)

    def act(self, board: np.ndarray) -> int:
        if self._rng is None:
            self.mount()
        rng    = self._rng
        qtable = self._qtable

        player = current_player(board)
        opp    = -player
        legal  = get_free_cols(board)

        # --- Respuesta inmediata: ganar o bloquear ---
        for col in legal:
            if check_winner(apply_move(board, col, player), player):
                return col
        for col in legal:
            if check_winner(apply_move(board, col, opp), opp):
                return col

        # --- Features del estado actual para actualizar la Q-table ---
        features = board_features(board, player)

        # --- Fase TBOPI: trials con UCB + LGRF ---
        lgrf_table: LGRFTable         = {}
        q_local: dict[int, float]     = {a: 0.0 for a in legal}
        n_local: dict[int, int]       = {a: 0   for a in legal}
        total_n = 0

        for _ in range(self.n_trials):
            a = _ucb_select(q_local, n_local, legal, total_n, c=self.ucb_c)
            reward = run_trial(
                apply_move(board, a, player),
                player,
                lgrf_table,
                qtable,
                rng,
                rollout_eps=self.rollout_eps,
            )
            total_n    += 1
            n_local[a] += 1
            q_local[a] += (reward - q_local[a]) / n_local[a]  # Welford

            # Actualizar Q-table persistente con cada trial
            _update_qtable(qtable, player, features, a, reward, alpha=self.qtable_alpha)

        # Persistir el conocimiento acumulado en este turno
        save_qtable(qtable, self.qtable_path)

        # --- Paso PImp: softmax sobre Q local para samplear acción final ---
        scores = np.array([max(q_local[a], 0.0) + 1e-6 for a in legal])
        scores = np.exp((scores - scores.max()) / self.temperature)
        probs  = scores / scores.sum()
        return int(rng.choice(legal, p=probs))