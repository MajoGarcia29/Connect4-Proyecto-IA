import math
import time
import random
import os
import pickle
import numpy as np
from collections import deque
from connect4.policy import Policy
from connect4.connect_state import ConnectState


# ============================================================
# Funciones Helper
# ============================================================

def _find_row(board, col):
    """Encuentra la fila donde caería una ficha en la columna dada.
    
    Args:
        board: Tablero numpy 6x7
        col: Índice de columna (0-6)
    
    Returns:
        Índice de fila (0-5), o -1 si la columna está llena.
    """
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            return r
    return -1


def can_win(board, col, player):
    """Detecta si colocar una ficha de 'player' en 'col' gana la partida.
    
    Coloca temporalmente la ficha, verifica las 4 direcciones,
    y revierte el cambio. No modifica el tablero permanentemente.
    
    Args:
        board: Tablero numpy 6x7
        col: Columna donde jugar (0-6)
        player: Jugador (-1 o 1)
    
    Returns:
        True si el movimiento gana, False en caso contrario.
    """
    row = _find_row(board, col)
    if row == -1:
        return False

    ROWS, COLS = 6, 7

    # Colocar ficha temporalmente
    board[row, col] = player
    won = False

    # Horizontal
    count = 1
    c = col - 1
    while c >= 0 and board[row, c] == player: count += 1; c -= 1
    c = col + 1
    while c < COLS and board[row, c] == player: count += 1; c += 1
    if count >= 4:
        won = True

    # Vertical
    if not won:
        count = 1
        r = row + 1
        while r < ROWS and board[r, col] == player: count += 1; r += 1
        if count >= 4:
            won = True

    # Diagonal ↘
    if not won:
        count = 1
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0 and board[r, c] == player: count += 1; r -= 1; c -= 1
        r, c = row + 1, col + 1
        while r < ROWS and c < COLS and board[r, c] == player: count += 1; r += 1; c += 1
        if count >= 4:
            won = True

    # Diagonal ↗
    if not won:
        count = 1
        r, c = row - 1, col + 1
        while r >= 0 and c < COLS and board[r, c] == player: count += 1; r -= 1; c += 1
        r, c = row + 1, col - 1
        while r < ROWS and c >= 0 and board[r, c] == player: count += 1; r += 1; c -= 1
        if count >= 4:
            won = True

    # Revertir ficha temporal
    board[row, col] = 0
    return won


def check_winner_after_move(board, col, player):
    """Verifica si 'player' ganaría tras jugar en 'col'.
    
    Wrapper semántico sobre can_win().
    No modifica el tablero.
    
    Args:
        board: Tablero numpy 6x7
        col: Columna donde jugar (0-6)
        player: Jugador (-1 o 1)
    
    Returns:
        player si gana, 0 en caso contrario.
    """
    return player if can_win(board, col, player) else 0


def hash_board(board):
    """Convierte el tablero numpy a una tupla hasheable.
    
    Usado como clave de diccionario para los Q-values.
    Sin colisiones: representación 1-a-1 del estado.
    
    Args:
        board: Tablero numpy 6x7
    
    Returns:
        Tupla de 42 enteros.
    """
    return tuple(board.flatten().tolist())


# ============================================================
# Smart Fast Rollout
# ============================================================

# Orden de preferencia de columnas: centro > bordes
_CENTER_PRIORITY = [3, 2, 4, 1, 5, 0, 6]


def smart_fast_rollout(board, player, q_values=None):
    """Simulación rápida con heurísticas inteligentes.
    
    Prioridades en cada turno del rollout:
      1. Ganar si es posible en 1 movimiento
      2. Bloquear victoria inmediata del oponente
      3. Usar Q-values pre-entrenados si están disponibles
      4. Preferir columnas centrales sobre bordes
      5. Fallback: movimiento aleatorio
    
    Args:
        board: Tablero numpy 6x7
        player: Jugador actual (-1 o 1)
        q_values: Dict opcional {board_hash: {col: q_value}}
    
    Returns:
        Ganador (-1, 1) o 0 (empate).
    """
    b = board.copy()
    ROWS, COLS = 6, 7
    p = player

    while True:
        # Columnas disponibles
        free_cols = [c for c in range(COLS) if b[0, c] == 0]
        if not free_cols:
            return 0  # Empate

        move = None

        # PRIORIDAD 1: ¿Puedo ganar ahora?
        for c in free_cols:
            if can_win(b, c, p):
                move = c
                break

        # PRIORIDAD 2: ¿Puede ganar el oponente? → Bloquear
        if move is None:
            for c in free_cols:
                if can_win(b, c, -p):
                    move = c
                    break

        # PRIORIDAD 3: Usar Q-values pre-entrenados
        if move is None and q_values is not None:
            bkey = hash_board(b)
            if bkey in q_values:
                col_q = q_values[bkey]
                # Elegir la columna libre con mayor Q-value
                valid = [(c, col_q[c]) for c in free_cols if c in col_q]
                if valid:
                    move = max(valid, key=lambda x: x[1])[0]

        # PRIORIDAD 4: Preferir columnas centrales
        if move is None:
            for c in _CENTER_PRIORITY:
                if c in free_cols:
                    move = c
                    break

        # FALLBACK: Movimiento aleatorio (seguridad, no debería llegar aquí
        # ya que _CENTER_PRIORITY cubre las 7 columnas)
        if move is None:
            move = random.choice(free_cols)

        # Aplicar movimiento
        for r in range(ROWS - 1, -1, -1):
            if b[r, move] == 0:
                b[r, move] = p
                row = r
                break

        # ── Verificar ganador (inline para máxima eficiencia) ──

        # Horizontal
        count = 1
        c = move - 1
        while c >= 0 and b[row, c] == p: count += 1; c -= 1
        c = move + 1
        while c < COLS and b[row, c] == p: count += 1; c += 1
        if count >= 4: return p

        # Vertical
        count = 1
        r = row + 1
        while r < ROWS and b[r, move] == p: count += 1; r += 1
        if count >= 4: return p

        # Diagonal ↘
        count = 1
        r, c = row - 1, move - 1
        while r >= 0 and c >= 0 and b[r, c] == p: count += 1; r -= 1; c -= 1
        r, c = row + 1, move + 1
        while r < ROWS and c < COLS and b[r, c] == p: count += 1; r += 1; c += 1
        if count >= 4: return p

        # Diagonal ↗
        count = 1
        r, c = row - 1, move + 1
        while r >= 0 and c < COLS and b[r, c] == p: count += 1; r -= 1; c += 1
        r, c = row + 1, move - 1
        while r < ROWS and c >= 0 and b[r, c] == p: count += 1; r += 1; c -= 1
        if count >= 4: return p

        p = -p


# ============================================================
# MCTS Node
# ============================================================

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

    def best_child(self, c_param, pretrained_q_values=None):
        """Selecciona el mejor hijo usando UCB1.
        
        Si pretrained_q_values está disponible, los nodos no visitados
        se ordenan por su Q-value pre-entrenado en lugar de explorar
        en orden arbitrario. Los Q-values actúan como "predicción inicial"
        que se reemplaza por datos reales tras la primera visita.
        
        Args:
            c_param: Constante de exploración (típicamente sqrt(2))
            pretrained_q_values: Dict opcional {board_hash: {col: q_value}}
        
        Returns:
            MCTSNode hijo con mayor peso UCB1.
        """
        # Obtener Q-values del estado padre si están disponibles
        prior_values = {}
        if pretrained_q_values:
            parent_hash = hash_board(self.state.board)
            prior_values = pretrained_q_values.get(parent_hash, {})

        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                if prior_values and child.action in prior_values:
                    # Usar Q-value como predicción inicial del nodo.
                    # Base 1e8 asegura que nodos no visitados se exploren
                    # antes de re-visitar nodos ya visitados (UCB << 1e8).
                    # El Q-prior (entre -1 y 1) ordena la exploración
                    # entre nodos no visitados: mejores primero.
                    weight = 1e8 + prior_values[child.action]
                else:
                    # Sin prior: prioridad neutral (base sin sesgo)
                    weight = 1e8
            else:
                q_hat = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = q_hat + exploration
            choices_weights.append(weight)

        max_weight = max(choices_weights)
        best_children = [self.children[i] for i, w in enumerate(choices_weights) if w == max_weight]
        return random.choice(best_children)


# ============================================================
# MCTS Agent
# ============================================================

class MCTSAgent(Policy):
    def __init__(self, iterations=10000, c_param=math.sqrt(2), pretrained_q_values=None):
        """Agente MCTS con soporte para Q-values pre-entrenados.
        
        Args:
            iterations: Número máximo de iteraciones MCTS por turno.
            c_param: Constante de exploración UCB1 (default: sqrt(2)).
            pretrained_q_values: Dict {board_hash: {col: q_value}} o None.
                Si es None, intenta cargar automáticamente desde
                'q_values.pkl' en el mismo directorio del script.
        """
        self.iterations = iterations
        self.c_param = c_param
        self.pretrained_q_values = pretrained_q_values
        self.last_root = None
        self.timeout = None

        # Auto-cargar Q-values si existe archivo junto al script
        if self.pretrained_q_values is None:
            try:
                _auto_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'q_values.pkl'
                )
                if os.path.exists(_auto_path):
                    self.pretrained_q_values = load_q_values(_auto_path)
            except Exception:
                pass

    def mount(self, timeout: float = None, *args, **kwargs) -> None:
        self.timeout = timeout

    def get_action(self, state: ConnectState) -> int:
        """Ejecuta MCTS y retorna la mejor acción.
        
        Args:
            state: Estado actual del juego.
        
        Returns:
            Índice de columna (0-6) para jugar.
        """
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

            # 1. Selección — desciende por el árbol usando UCB1 + priors
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.c_param, self.pretrained_q_values)

            # 2. Expansión
            if not node.is_terminal():
                node = node.expand()

            # 3. Simulación — usa smart rollout con Q-values
            if node.is_terminal():
                winner = node.state.get_winner()
            else:
                winner = smart_fast_rollout(
                    node.state.board, node.state.player, self.pretrained_q_values
                )

            # 4. Backpropagación
            while node is not None:
                node.visits += 1
                if winner != 0:
                    player_who_just_moved = -node.state.player
                    if winner == player_who_just_moved:
                        node.value += 1.0
                    else:
                        node.value -= 1.0
                node = node.parent

        # Guardar referencia al árbol para extracción posterior de Q-values
        self.last_root = root

        if not root.children:
            free_cols = state.get_free_cols()
            return random.choice(free_cols) if free_cols else 0

        # Elegir acción con más visitas (más robusto que max q-value)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def act(self, s: np.ndarray) -> int:
        player = -1 if np.sum(s) == 0 else 1
        state = ConnectState(s, player)
        return self.get_action(state)


# ============================================================
# Q-Values: Extracción, Fusión, Guardado y Carga
# ============================================================

def extract_q_values(root_node):
    """Extrae Q-values del árbol MCTS usando BFS.
    
    Recorre el árbol desde la raíz y, para cada nodo hijo
    con al menos una visita, registra su q-hat (value/visits)
    como el Q-value de la acción correspondiente desde el padre.
    
    Args:
        root_node: Nodo raíz del árbol MCTS.
    
    Returns:
        Dict {board_hash: {col: q_value}} con los Q-values extraídos.
    """
    q_values = {}
    queue = deque([root_node])

    while queue:
        node = queue.popleft()
        for child in node.children:
            if child.visits > 0:
                board_key = hash_board(node.state.board)
                if board_key not in q_values:
                    q_values[board_key] = {}
                q_values[board_key][child.action] = child.value / child.visits
                queue.append(child)

    return q_values


def merge_q_values(existing, new):
    """Fusiona Q-values nuevos con existentes promediando duplicados.
    
    Para estados/acciones que existen en ambos diccionarios,
    promedia los valores. Para nuevos, los agrega directamente.
    Modifica 'existing' in-place y lo retorna.
    
    Args:
        existing: Dict de Q-values acumulados (se modifica in-place).
        new: Dict de Q-values nuevos a fusionar.
    
    Returns:
        El dict 'existing' actualizado.
    """
    for board_key, actions in new.items():
        if board_key not in existing:
            existing[board_key] = {}
        for action, value in actions.items():
            if action in existing[board_key]:
                # Promediar con el valor existente
                existing[board_key][action] = (existing[board_key][action] + value) / 2
            else:
                existing[board_key][action] = value
    return existing


def save_q_values(agent_or_dict, filepath):
    """Guarda Q-values a archivo pickle.
    
    Acepta un MCTSAgent (extrae Q-values de su último árbol)
    o un dict de Q-values directamente.
    
    Args:
        agent_or_dict: MCTSAgent o dict {board_hash: {col: q_value}}.
        filepath: Ruta del archivo pickle de destino.
    
    Raises:
        ValueError: Si el agente no ha jugado ninguna partida.
        TypeError: Si el argumento no es MCTSAgent ni dict.
    """
    if isinstance(agent_or_dict, MCTSAgent):
        if agent_or_dict.last_root is None:
            raise ValueError(
                "El agente no ha jugado ninguna partida aún (last_root es None)"
            )
        q_values = extract_q_values(agent_or_dict.last_root)
    elif isinstance(agent_or_dict, dict):
        q_values = agent_or_dict
    else:
        raise TypeError("Se esperaba MCTSAgent o dict")

    with open(filepath, 'wb') as f:
        pickle.dump(q_values, f)


def load_q_values(filepath):
    """Carga Q-values desde archivo pickle.
    
    Args:
        filepath: Ruta del archivo pickle.
    
    Returns:
        Dict de Q-values, o None si el archivo no existe.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)