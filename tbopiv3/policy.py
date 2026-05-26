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

import math
import pickle
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
# Features del tablero para la Q-table persistente
# ---------------------------------------------------------------------------

def _count_open_threats(board: np.ndarray, player: int, length: int) -> int:
    """
    Cuenta secuencias de `length` fichas de `player` con al menos
    un extremo libre. Estas son amenazas reales que pueden completarse.
    """
    count = 0
    # Horizontal
    for r in range(6):
        for c in range(7 - length + 1):
            if np.all(board[r, c:c+length] == player):
                if (c > 0 and board[r, c-1] == 0) or (c + length < 7 and board[r, c+length] == 0):
                    count += 1
    # Vertical
    for c in range(7):
        for r in range(6 - length + 1):
            if np.all(board[r:r+length, c] == player):
                if r > 0 and board[r-1, c] == 0:
                    count += 1
    # Diagonales
    for r in range(6 - length + 1):
        for c in range(7 - length + 1):
            # Abajo-derecha
            if all(board[r+i, c+i] == player for i in range(length)):
                if (r > 0 and c > 0 and board[r-1, c-1] == 0) or \
                   (r + length < 6 and c + length < 7 and board[r+length, c+length] == 0):
                    count += 1
            # Arriba-derecha (ajustado para c >= length-1)
            c_adj = c + length - 1
            if all(board[r+i, c_adj-i] == player for i in range(length)):
                if (r > 0 and c_adj < 6 and board[r-1, c_adj+1] == 0) or \
                   (r + length < 6 and c_adj >= length and board[r+length, c_adj-length] == 0):
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
    own_threats = _count_open_threats(board, player, length=3)
    opp_threats = _count_open_threats(board, -player, length=3)
    center_ctrl = int(np.sum(board[:, 2:5] == player))

    return (own_threats, opp_threats, center_ctrl)


# ---------------------------------------------------------------------------
# Q-table persistente (aprende entre partidas)
# ---------------------------------------------------------------------------

# Tipo para la Q-table: (player, own_t, opp_t, center_c, col) -> float
QTable = dict[tuple[int, int, int, int, int], float]

# --- Persistencia ---

def load_qtable(path: Path) -> QTable:
    """Carga la Q-table desde disco. Devuelve tabla vacía si no existe o está vacía."""
    try:
        if not path.exists() or path.stat().st_size < 10:
            return {}
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def save_qtable(table: QTable, path: Path) -> None:
    """
    Guarda la Q-table fusionándola con el conocimiento previo en el disco.
    Usa un archivo temporal (.tmp) para garantizar que el archivo .pkl nunca
    quede vacío o corrupto si la escritura falla o se interrumpe.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Fusionar con lo que ya existe en disco (prevenir pérdida de datos)
        data = load_qtable(path)
        data.update(table)
        
        # 2. Escritura atómica: primero a temporal, luego renombrar
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
    except Exception:
        # Fallback: si falla el renombrado, intentar escritura directa
        with open(path, "wb") as f:
            pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---------------------------------------------------------------------------
# Tabla LGRF (Last Good Reply with Forgetting)
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
# Política de rollout: LGRF → Q-table → random
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
        q_vals = {c: qtable.get((player, *features, c), 0.0) for c in free}
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
    Rollout hasta estado terminal.

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

def ucb_select(
    q_local: dict[int, float],
    n_local: dict[int, int],
    legal: list[int],
    total_n: int,
    c: float = math.sqrt(2),
    log_n: float = 0.0,
) -> int:
    """
    Elige la acción con mayor valor UCB.
    """
    # Priorizar no visitadas con orden preferente al centro: [3, 2, 4, 1, 5, 0, 6]
    for a in [3, 2, 4, 1, 5, 0, 6]:
        if a in legal and n_local.get(a, 0) == 0:
            return a
    
    best_action, best_score = legal[0], -math.inf
    for a in legal:
        n_a = n_local[a]
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
    qtable_path:
        Ruta al archivo JSON donde se persiste la Q-table entre partidas.
    qtable_alpha:
        Tasa de aprendizaje para actualizar la Q-table (TD(0)).
    """

    def __init__(
        self,
        n_trials: int     = 200,
        seed: int         = 42,
        ucb_c: float      = 1.414,
        rollout_eps: float = 0.3,
        qtable_path: str | None = None,
        qtable_alpha: float = 0.1,
    ) -> None:
        self.n_trials     = n_trials
        self.seed         = seed
        self.ucb_c        = ucb_c
        self.rollout_eps  = rollout_eps
        self.qtable_alpha = qtable_alpha
        self._rng: np.random.RandomState | None = None
        self._qtable: QTable = {}

        # Si no se define ruta, guardar en la carpeta del grupo por defecto
        self.qtable_path = Path(qtable_path) if qtable_path else Path(__file__).parent / "qtable.pkl"

    def mount(self, _timeout: float | None = None) -> None:
        self._rng    = np.random.RandomState(self.seed)
        self._qtable = load_qtable(self.qtable_path)

    def save_qtable(self) -> None:
        """Persiste la Q-table actual en el disco."""
        save_qtable(self._qtable, self.qtable_path)
        # Sincronizar memoria con el disco tras la fusión
        self._qtable = load_qtable(self.qtable_path)

    def act(self, board: np.ndarray) -> int:
        if self._rng is None:
            self.mount()
        rng    = self._rng
        qtable = self._qtable

        p, opp = current_player(board), -current_player(board)
        legal  = get_free_cols(board)

        # --- Respuesta inmediata: ganar o bloquear ---
        for col in legal:
            if check_winner(apply_move(board, col, p), p): return col
            if check_winner(apply_move(board, col, opp), opp): return col

        # --- Features del estado actual para actualizar la Q-table ---
        feats = board_features(board, p)

        # --- Ciclo de Entrenamiento: Trials con UCB + LGRF ---
        lgrf_table: LGRFTable         = {}
        q_local: dict[int, float]     = {a: 0.0 for a in legal}
        n_local: dict[int, int]       = {a: 0   for a in legal}
        total_n = 0

        for _ in range(self.n_trials):
            log_n = math.log(total_n) if total_n > 0 else 0.0
            a = ucb_select(q_local, n_local, legal, total_n, self.ucb_c, log_n)
            reward = run_trial(
                apply_move(board, a, p), p, lgrf_table, qtable, rng, self.rollout_eps
            )
            total_n    += 1
            n_local[a] += 1
            q_local[a] += (reward - q_local[a]) / n_local[a]  # Welford

            # Actualizar Q-table persistente con cada trial
            # Q(s, a) <- Q(s, a) + alpha * (reward - Q(s, a))
            key = (p, *feats, a)
            q_old = qtable.get(key, 0.0)
            qtable[key] = q_old + self.qtable_alpha * (reward - q_old)

        # Persistencia inmediata para asegurar que el entrenamiento capture cada mejora
        self.save_qtable()

        # --- Paso PImp: selección greedy sobre Q local ---
        best_q = max(q_local.values())
        best_actions = [a for a in legal if q_local[a] == best_q]
        return int(rng.choice(best_actions))