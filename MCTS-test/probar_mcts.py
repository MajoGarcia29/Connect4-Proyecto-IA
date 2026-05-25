"""
probar_mcts.py
--------------
Prueba el MCTS guiado por Q-values de Majo contra el agente FVMC.
Version standalone - no depende del framework de Gradescope.

Uso:
    python probar_mcts.py

IMPORTANTE: necesitas tener qvalues_majo.pkl en la misma carpeta.
"""

import pickle
import random
import numpy as np

# ── Utilidades de tablero ─────────────────────────────────────────────────────

def jugador_actual(board):
    return -1 if np.sum(board == -1) == np.sum(board == 1) else 1

def acciones_legales(board):
    return [c for c in range(7) if board[0, c] == 0]

def aplicar_accion(board, col, player):
    nuevo = board.copy()
    for fila in reversed(range(6)):
        if nuevo[fila, col] == 0:
            nuevo[fila, col] = player
            break
    return nuevo

def hay_ganador(board, player):
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

def es_terminal(board):
    return (
        hay_ganador(board, -1)
        or hay_ganador(board, 1)
        or len(acciones_legales(board)) == 0
    )

def recompensa_final(board, mi_jugador):
    if hay_ganador(board, mi_jugador):
        return 1.0
    if hay_ganador(board, -mi_jugador):
        return -1.0
    return 0.0

def key(board):
    return board.tobytes()

# ── TablaQ ────────────────────────────────────────────────────────────────────

class TablaQ:
    def __init__(self):
        self._q = {}
        self._n = {}

    def get(self, board, accion):
        return self._q.get(key(board), {}).get(accion, 0.0)

    def update(self, board, accion, U):
        k = key(board)
        if k not in self._q:
            self._q[k] = {}
            self._n[k] = {}
        if accion not in self._q[k]:
            self._q[k][accion] = 0.0
            self._n[k][accion] = 0
        self._n[k][accion] += 1
        n = self._n[k][accion]
        self._q[k][accion] += (U - self._q[k][accion]) / n

    def mejor_accion(self, board, acciones):
        vals = self._q.get(key(board), {})
        return max(acciones, key=lambda a: vals.get(a, 0.0))

# ── Agente FVMC de Majo ───────────────────────────────────────────────────────

class FVMCPolicy:
    def __init__(self, n_partidas=5000, epsilon=0.1):
        self.n_partidas = n_partidas
        self.epsilon = epsilon
        self._q = TablaQ()
        self._rng = np.random.RandomState(42)

    def train(self):
        for i in range(self.n_partidas):
            if i % 500 == 0:
                print(f"  Partida {i}/{self.n_partidas}...")
            trayectoria = self._jugar_partida()
            self._actualizar_q(trayectoria)

    def act(self, board):
        acciones = acciones_legales(board)
        if not acciones:
            return None
        jugador = jugador_actual(board)
        for a in acciones:
            if hay_ganador(aplicar_accion(board, a, jugador), jugador):
                return a
        oponente = -jugador
        for a in acciones:
            if hay_ganador(aplicar_accion(board, a, oponente), oponente):
                return a
        vals = [self._q.get(board, a) for a in acciones]
        if max(vals) == 0:
            return int(self._rng.choice(acciones))
        return self._q.mejor_accion(board, acciones)

    def _elegir_accion(self, board, acciones):
        if self._rng.rand() < self.epsilon:
            return int(self._rng.choice(acciones))
        return self._q.mejor_accion(board, acciones)

    def _jugar_partida(self):
        board = np.zeros((6, 7), dtype=int)
        trayectoria = []
        while not es_terminal(board):
            acciones = acciones_legales(board)
            jugador = jugador_actual(board)
            accion = self._elegir_accion(board, acciones)
            trayectoria.append((board.copy(), accion, jugador))
            board = aplicar_accion(board, accion, jugador)
        trayectoria.append((board.copy(), None, None))
        return trayectoria

    def _actualizar_q(self, trayectoria):
        board_terminal = trayectoria[-1][0]
        mi_jugador = trayectoria[0][2]
        U = recompensa_final(board_terminal, mi_jugador)
        vistos = set()
        for board, accion, jugador in reversed(trayectoria[:-1]):
            if jugador != mi_jugador:
                U = -U
            par = (key(board), accion)
            if par not in vistos:
                vistos.add(par)
                self._q.update(board, accion, U)

# ── Rollout guiado por Q-values ───────────────────────────────────────────────

def rollout_qguiado(board, player, q_table, epsilon_rollout=0.2):
    b = board.copy()
    p = player

    while True:
        free_cols = acciones_legales(b)
        if not free_cols:
            return 0

        move = None
        if q_table and random.random() > epsilon_rollout:
            k = key(b)
            if k in q_table:
                candidatos = {a: v for a, v in q_table[k].items() if a in free_cols}
                if candidatos and max(candidatos.values()) > 0:
                    move = max(candidatos, key=candidatos.get)

        if move is None:
            move = random.choice(free_cols)

        b = aplicar_accion(b, move, p)

        if hay_ganador(b, p):
            return p

        p = -p

# ── Agente MCTS guiado (standalone) ──────────────────────────────────────────

import math

class MCTSNode:
    def __init__(self, board, player, parent=None, action=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = acciones_legales(board)
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return es_terminal(self.board)

    def expand(self):
        action = self.untried_actions.pop()
        next_board = aplicar_accion(self.board, action, self.player)
        next_player = -self.player
        child = MCTSNode(next_board, next_player, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, c_param):
        weights = []
        for child in self.children:
            if child.visits == 0:
                w = float('inf')
            else:
                w = child.value / child.visits + c_param * math.sqrt(math.log(self.visits) / child.visits)
            weights.append(w)
        max_w = max(weights)
        best = [self.children[i] for i, w in enumerate(weights) if w == max_w]
        return random.choice(best)


class MCTSQGuiadoStandalone:
    def __init__(self, q_table, iterations=500, epsilon_rollout=0.2):
        self.q_table = q_table
        self.iterations = iterations
        self.epsilon_rollout = epsilon_rollout

    def act(self, board):
        player = jugador_actual(board)
        root = MCTSNode(board.copy(), player)

        for _ in range(self.iterations):
            node = root

            # Selección
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(math.sqrt(2))

            # Expansión
            if not node.is_terminal():
                node = node.expand()

            # Simulación
            if node.is_terminal():
                winner = 1 if hay_ganador(node.board, 1) else (-1 if hay_ganador(node.board, -1) else 0)
            else:
                winner = rollout_qguiado(node.board, node.player, self.q_table, self.epsilon_rollout)

            # Backpropagación
            while node is not None:
                node.visits += 1
                if winner != 0:
                    player_moved = -node.player
                    if winner == player_moved:
                        node.value += 1.0
                    else:
                        node.value -= 1.0
                node = node.parent

        if not root.children:
            return random.choice(acciones_legales(board))

        return max(root.children, key=lambda c: c.visits).action

# ── Jugar partidas ────────────────────────────────────────────────────────────

def jugar_partida(agente1, agente2, verbose=False):
    board = np.zeros((6, 7), dtype=int)
    turno = 1

    while True:
        libres = acciones_legales(board)
        if not libres:
            return 0

        col = agente1.act(board) if turno == 1 else agente2.act(board)
        if col not in libres:
            col = random.choice(libres)

        board = aplicar_accion(board, col, turno)

        if verbose:
            print(board)
            print()

        if hay_ganador(board, turno):
            return turno

        turno = -turno


def torneo(agente1, nombre1, agente2, nombre2, n=20):
    v1 = v2 = empates = 0
    for i in range(n):
        if i % 2 == 0:
            r = jugar_partida(agente1, agente2)
        else:
            r = jugar_partida(agente2, agente1)
            r = -r if r != 0 else 0

        if r == 1:   v1 += 1
        elif r == -1: v2 += 1
        else:         empates += 1

        print(f"  Partida {i+1}/{n} — resultado: {'empate' if r==0 else (nombre1 if r==1 else nombre2)}")

    print(f"\n{'='*40}")
    print(f"  {nombre1}: {v1} victorias")
    print(f"  {nombre2}: {v2} victorias")
    print(f"  Empates:  {empates}")
    print(f"{'='*40}")
    ganador = nombre1 if v1 > v2 else (nombre2 if v2 > v1 else "Empate")
    print(f"  Ganador: {ganador}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Cargar Q-values de Majo
    print("Cargando Q-values de Majo...")
    with open("qvalues_majo.pkl", "rb") as f:
        data = pickle.load(f)
    q_table = data["q"]
    print(f"Estados cargados: {len(q_table):,}\n")

    # 2. Crear agente de Majo (carga los Q-values, no re-entrena)
    print("Inicializando agente de Majo...")
    majo = FVMCPolicy(n_partidas=0)
    majo._q._q = data["q"]
    majo._q._n = data["n"]

    # 3. Crear agente MCTS guiado de Juan Diego
    print("Inicializando MCTS guiado...\n")
    juandi = MCTSQGuiadoStandalone(q_table=q_table, iterations=200, epsilon_rollout=0.2)

    # 4. Torneo
    print("Iniciando torneo: Majo vs Juan Diego (MCTS guiado)")
    print("="*40)
    torneo(majo, "Majo (FVMC)", juandi, "JuanDi (MCTS+Q)", n=20)