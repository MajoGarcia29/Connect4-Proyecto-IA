"""
mcts_qguiado.py
---------------
Agente de Juan Diego (MCTS) mejorado con los Q-values de Majo.

Cambios respecto al MCTS original:
  1. Carga qvalues_majo.pkl al iniciar.
  2. En la fase de SIMULACIÓN (rollout), en vez de elegir columnas
     totalmente al azar, consulta los Q-values de Majo:
       - Si el estado está en la tabla → elige la mejor acción conocida
         con probabilidad (1 - epsilon_rollout), aleatorio si no.
       - Si el estado NO está en la tabla → aleatorio puro (igual que antes).
  Esto hace que los rollouts sean mucho más informativos y el árbol
  converja más rápido a buenas jugadas.
"""

import math
import time
import random
import pickle
import numpy as np

from connect4.policy import Policy
from connect4.connect_state import ConnectState


# ──────────────────────────────────────────────
#  Utilidades de tablero
# ──────────────────────────────────────────────

def _key(board: np.ndarray) -> bytes:
    """Misma función key() que usa Majo — imprescindible para hacer match."""
    return board.tobytes()


def _hay_ganador(board: np.ndarray, player: int) -> bool:
    for r in range(6):
        for c in range(7):
            if board[r, c] != player:
                continue
            if c + 3 < 7 and all(board[r, c+i] == player for i in range(4)):
                return True
            if r + 3 < 6 and all(board[r+i, c] == player for i in range(4)):
                return True
            if r + 3 < 6 and c + 3 < 7 and all(board[r+i, c+i] == player for i in range(4)):
                return True
            if r + 3 < 6 and c - 3 >= 0 and all(board[r+i, c-i] == player for i in range(4)):
                return True
    return False


def fast_rollout_qguiado(
    board: np.ndarray,
    player: int,
    q_table: dict,
    epsilon_rollout: float = 0.2,
) -> int:
    """
    Rollout guiado por Q-values de Majo.

    - Si el estado está en q_table y hay valores > 0:
        con prob (1-epsilon_rollout) elige la acción con mayor Q-value.
    - En cualquier otro caso: movimiento aleatorio.

    Retorna: jugador ganador (1 o -1) o 0 si empate.
    """
    b = board.copy()
    ROWS, COLS = 6, 7
    p = player

    while True:
        free_cols = [c for c in range(COLS) if b[0, c] == 0]
        if not free_cols:
            return 0  # empate

        # ── Intentar usar Q-values de Majo ──
        move = None
        if q_table and random.random() > epsilon_rollout:
            k = _key(b)
            if k in q_table:
                q_acciones = q_table[k]
                # Filtramos solo columnas legales
                candidatos = {a: v for a, v in q_acciones.items() if a in free_cols}
                if candidatos and max(candidatos.values()) > 0:
                    move = max(candidatos, key=candidatos.get)

        if move is None:
            move = random.choice(free_cols)

        # ── Aplicar movimiento ──
        for r in range(ROWS - 1, -1, -1):
            if b[r, move] == 0:
                b[r, move] = p
                break

        # ── Verificar ganador ──
        if _hay_ganador(b, p):
            return p

        p = -p


# ──────────────────────────────────────────────
#  Nodo MCTS (igual que el original de Juan Diego)
# ──────────────────────────────────────────────

class MCTSNode:
    def __init__(self, state: ConnectState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = state.get_free_cols()
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state.is_final()

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.transition(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                q_hat = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = q_hat + exploration
            choices_weights.append(weight)

        max_weight = max(choices_weights)
        best_children = [
            self.children[i]
            for i, w in enumerate(choices_weights)
            if w == max_weight
        ]
        return random.choice(best_children)


# ──────────────────────────────────────────────
#  Agente MCTS + Q-values de Majo
# ──────────────────────────────────────────────

class MCTSQGuiadoAgent(Policy):
    """
    MCTS de Juan Diego con rollouts guiados por los Q-values de Majo.

    Parámetros
    ----------
    q_path : str
        Ruta al archivo .pkl generado por guardar_qvalues.py
    iterations : int
        Iteraciones MCTS por jugada (igual que el original).
    epsilon_rollout : float
        Prob. de movimiento aleatorio dentro del rollout (0 = siempre Q-guiado,
        1 = siempre aleatorio). 0.2 es un buen balance.
    """

    def __init__(
        self,
        q_path: str = "qvalues_majo.pkl",
        iterations: int = 10000,
        c_param: float = math.sqrt(2),
        epsilon_rollout: float = 0.2,
    ):
        self.iterations = iterations
        self.c_param = c_param
        self.epsilon_rollout = epsilon_rollout
        self.timeout = None

        # Cargar Q-values de Majo
        self._q_table = {}
        try:
            with open(q_path, "rb") as f:
                data = pickle.load(f)
            self._q_table = data["q"]   # dict: key(board) -> {col: q_value}
            print(f"[MCTSQGuiado] Q-values cargados: {len(self._q_table):,} estados")
        except FileNotFoundError:
            print(f"[MCTSQGuiado] ADVERTENCIA: no se encontró '{q_path}'.")
            print("  Corre primero: python guardar_qvalues.py")
            print("  El agente funcionará con rollouts aleatorios puros.")

    def mount(self, timeout: float = None, *args, **kwargs) -> None:
        self.timeout = timeout

    def get_action(self, state: ConnectState) -> int:
        free_cols = state.get_free_cols()
        if not free_cols:
            return 0

        root = MCTSNode(state)
        start_time = time.time()
        time_limit = (self.timeout * 0.9) if self.timeout else None

        for _ in range(self.iterations):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            node = root

            # 1. Selección
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.c_param)

            # 2. Expansión
            if not node.is_terminal():
                node = node.expand()

            # 3. Simulación (rollout guiado por Q-values de Majo)
            if node.is_terminal():
                winner = node.state.get_winner()
            else:
                winner = fast_rollout_qguiado(
                    node.state.board,
                    node.state.player,
                    self._q_table,
                    self.epsilon_rollout,
                )

            # 4. Backpropagación (igual que el original)
            while node is not None:
                node.visits += 1
                if winner != 0:
                    player_who_just_moved = -node.state.player
                    if winner == player_who_just_moved:
                        node.value += 1.0
                    else:
                        node.value -= 1.0
                node = node.parent

        if not root.children:
            return random.choice(state.get_free_cols() or [0])

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def act(self, s: np.ndarray) -> int:
        player = -1 if np.sum(s) == 0 else 1
        state = ConnectState(s, player)
        return self.get_action(state)