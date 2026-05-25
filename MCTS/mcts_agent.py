import math
import random
import pickle
import os
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override

transposition_table = {}

def get_node_from_transposition(state: ConnectState):
    key = state.board.tobytes()
    return transposition_table.get((key, state.player))

class MCTSNode:
    def __init__(self, state: ConnectState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Acción que llevó a este nodo
        self.children = []
        self.visits = 0
        self.value = 0.0
        self._untried_actions = state.get_free_cols()

        key = (state.board.tobytes(), state.player)
        transposition_table[key] = self

    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0

    def is_terminal(self) -> bool:
        # CORRECCIÓN: El método oficial del framework es is_final()
        return self.state.is_final()

    def expand(self):
        # 1. Identificar jugadas prioritarias (Killer Moves)
        prioritarias = []
        restantes = []
        
        # Oponente
        opp = -self.state.player
        
        for action in self._untried_actions:
            # ¿Esta jugada gana el juego para mi?
            next_s_win = self.state.transition(action)
            if next_s_win.get_winner() == self.state.player:
                prioritarias.append(action)
                continue
            
            # ¿Esta jugada bloquea una victoria del oponente?
            if next_s_win.get_winner() == opp: # Si al mover yo, el oponente gana, es un bloqueo
                prioritarias.append(action)
                continue
                
            restantes.append(action)
            
        # 2. Reordenar acciones: Prioritarias primero, el resto después
        acciones_ordenadas = prioritarias + restantes
        self._untried_actions = acciones_ordenadas
        
        # 3. Expandir la primera (ahora será siempre una killer move si existe)
        action = self._untried_actions.pop(0) # Sacamos la mejor de la lista
        next_state = self.state.transition(action)
        child = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, c_param=1.41, q_func=None, lambda_weight=0.5):
        choices_weights = []
        for child in self.children:
            # 1. Valor empírico del árbol (Winrate de las simulaciones actuales)
            mcts_value = child.value / child.visits
            
            # 2. Valor histórico de la tabla Q (Aprendizaje pasado)
            q_value = 0.0
            if q_func is not None:
                q_value = q_func(self.state.board, child.action)
                
            # Combinación PUCT (Promedio entre el árbol y la tabla Q)
            if q_func is not None:
                exploit = (1 - lambda_weight) * mcts_value + lambda_weight * q_value
            else:
                exploit = mcts_value
                
            # 3. Exploración UCB clásica
            explore = c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            
            choices_weights.append(exploit + explore)
            
        return self.children[np.argmax(choices_weights)]


class MCTSQAgent(Policy):
    def __init__(self, iterations=100, alpha=0.1, gamma=0.9, q_path=None, c_param=1.41):
            self.iterations = iterations
            self.alpha = alpha  
            self.gamma = gamma  
            self.c_param = c_param
            self.epsilon_rollout = 0.2  
            
            # 1. Obtener la ruta absoluta de la carpeta donde está este script (mcts_agent.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. Si no se pasa un q_path explícito, usar el de esta misma carpeta
            if q_path is None:
                self.q_path = os.path.join(current_dir, "qvalues_majo.pkl")
            else:
                self.q_path = q_path
            
            # Cargar la tabla Q
            try:
                with open(self.q_path, "rb") as f:
                    data = pickle.load(f)
                    self.q_table = data.get("q", data) if isinstance(data, dict) else data
            except FileNotFoundError:
                print(f"[MCTS] No se encontró el archivo en {self.q_path}. Iniciando tabla vacía.")
                self.q_table = {}

    def mount(self) -> None:
        """Método obligatorio requerido por la clase base Policy."""
        pass

    def _get_canonical_state_and_action(self, board: np.ndarray, action: int = None):
        """Genera una clave única (estado canónico) usando simetría de espejo."""
        board_bytes = board.tobytes()
        flipped_board = np.fliplr(flipped_board if 'flipped_board' in locals() else board)
        flipped_board = np.fliplr(board)
        flipped_bytes = flipped_board.tobytes()

        if flipped_bytes < board_bytes:
            canonical_key = flipped_bytes
            canonical_action = (6 - action) if action is not None else None
            is_flipped = True
        else:
            canonical_key = board_bytes
            canonical_action = action
            is_flipped = False

        return canonical_key, canonical_action, is_flipped

    def get_q(self, board: np.ndarray, action: int) -> float:
        state_key, canonical_action, _ = self._get_canonical_state_and_action(board, action)
        return self.q_table.get(state_key, {}).get(canonical_action, 0.0)

    def update_q(self, board: np.ndarray, action: int, reward: float, next_board: np.ndarray, is_terminal: bool):
        """Actualización online TD-error aplicando equivalencia por simetrías."""
        state_key, canonical_action, _ = self._get_canonical_state_and_action(board, action)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            
        current_q = self.q_table[state_key].get(canonical_action, 0.0)
        
        max_next_q = 0.0
        if not is_terminal:
            next_key, _, _ = self._get_canonical_state_and_action(next_board)
            if next_key in self.q_table and self.q_table[next_key]:
                max_next_q = max(self.q_table[next_key].values())
                
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][canonical_action] = new_q

    def save_q(self):
        """Guarda permanentemente el progreso en formato estructurado."""
        with open(self.q_path, "wb") as f:
            pickle.dump({"q": self.q_table}, f)

    def fast_rollout_qguiado(self, board: np.ndarray, player: int) -> int:
        """Simulación acelerada que explota los Q-values existentes."""
        b = board.copy()
        p = player
        ROWS, COLS = 6, 7
        
        while True:
            free_cols = [c for c in range(COLS) if b[0, c] == 0]
            if not free_cols:
                return 0 
            
            state_key, _, is_flipped = self._get_canonical_state_and_action(b)
            
            if random.random() < self.epsilon_rollout or state_key not in self.q_table or not self.q_table[state_key]:
                move = random.choice(free_cols)
            else:
                q_values_legales = {}
                for c in free_cols:
                    canonical_c = (6 - c) if is_flipped else c
                    q_values_legales[c] = self.q_table[state_key].get(canonical_c, 0.0)
                move = max(q_values_legales, key=q_values_legales.get)
            
            for r in range(ROWS - 1, -1, -1):
                if b[r, move] == 0:
                    b[r, move] = p
                    row = r
                    break
                    
            # Validaciones rápidas de terminalidad (evita instanciaciones costosas)
            count = 1
            c = move - 1
            while c >= 0 and b[row, c] == p: count += 1; c -= 1
            c = move + 1
            while c < COLS and b[row, c] == p: count += 1; c += 1
            if count >= 4: return p
            
            count = 1
            r = row + 1
            while r < ROWS and b[r, move] == p: count += 1; r += 1
            if count >= 4: return p
            
            count = 1
            r, c = row + 1, move + 1
            while r < ROWS and c < COLS and b[r, c] == p: count += 1; r += 1; c += 1
            r, c = row - 1, move - 1
            while r >= 0 and c >= 0 and b[r, c] == p: count += 1; r -= 1; c -= 1
            if count >= 4: return p
            
            count = 1
            r, c = row + 1, move - 1
            while r < ROWS and c >= 0 and b[r, c] == p: count += 1; r += 1; c -= 1
            r, c = row - 1, move + 1
            while r >= 0 and c < COLS and b[r, c] == p: count += 1; r -= 1; c += 1
            if count >= 4: return p
            
            p = -p

    def get_action(self, state: ConnectState) -> int:
        """Método obligatorio original que procesa la búsqueda basada en simulación."""
        root = MCTSNode(state)

        for _ in range(self.iterations):
            node = root
            # 1. Selection
            while node.is_fully_expanded() and not node.is_terminal():
                # Inyectamos la tabla Q directamente en las raíces del árbol MCTS
                node = node.best_child(self.c_param, q_func=self.get_q, lambda_weight=0.5)
                
            if not node.is_terminal():
                node = node.expand()
                
            if node.is_terminal():
                winner = node.state.get_winner()
            else:
                winner = self.fast_rollout_qguiado(node.state.board, node.state.player)
            
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

    @override
    def act(self, s: np.ndarray) -> int:
        state = ConnectState(s)
        
        # --- NUEVA OPTIMIZACIÓN: JUGADA GANADORA INMEDIATA ---
        # Si existe una acción que gana el juego en 1 turno, tómala ya.
        for col in state.get_free_cols():
            next_state = state.transition(col)
            if next_state.get_winner() == state.player:
                return col
        
        # Si no hay victoria inmediata, revisa si hay que bloquear al oponente
        opp = -state.player
        for col in state.get_free_cols():
            next_state = state.transition(col)
            if next_state.get_winner() == opp:
                return col
        # ----------------------------------------------------

        # Si no hay victoria ni peligro inmediato, ejecuta el MCTS
        return self.get_action(state)