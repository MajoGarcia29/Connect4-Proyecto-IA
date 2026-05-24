"""
guardar_qvalues.py
------------------
Entrena el agente de Majo y guarda los Q-values en disco.
Version standalone - no depende del framework de Gradescope.

Uso:
    python guardar_qvalues.py
"""

import pickle
import numpy as np

# ── Copiar aquí las funciones de game_utils.py ────────────────────────────────

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

# ── Agente FVMC (sin depender de connect4) ───────────────────────────────────

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

# ── Main ──────────────────────────────────────────────────────────────────────

N_PARTIDAS = 5000

print(f"Entrenando agente de Majo con {N_PARTIDAS} partidas...")
agente = FVMCPolicy(n_partidas=N_PARTIDAS, epsilon=0.1)
agente.train()

q_data = {
    "q": agente._q._q,
    "n": agente._q._n,
}

with open("qvalues_majo.pkl", "wb") as f:
    pickle.dump(q_data, f)

print(f"\nQ-values guardados en 'qvalues_majo.pkl'")
print(f"Estados aprendidos: {len(q_data['q']):,}")
