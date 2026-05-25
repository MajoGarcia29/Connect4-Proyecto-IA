import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


def key(board: np.ndarray) -> bytes:
    return board.tobytes()


# ── TablaQ ────────────────────────────────────────────────────────────────────

class TablaQ:

    def __init__(self):
        self._q = {}
        self._n = {}

    def get(self, board: np.ndarray, accion: int) -> float:
        return self._q.get(key(board), {}).get(accion, 0.0)

    def update(self, board: np.ndarray, accion: int, U: float) -> None:
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

    def mejor_accion(self, board: np.ndarray, acciones: list) -> int:
        vals = self._q.get(key(board), {})
        return max(acciones, key=lambda a: vals.get(a, 0.0))


# ── FVMCPolicy ────────────────────────────────────────────────────────────────

class FVMCPolicy(Policy):

    def __init__(self, n_partidas: int = 5000, epsilon: float = 0.1):
        self.n_partidas = n_partidas
        self.epsilon = epsilon
        self._q = TablaQ()
        self._rng = np.random.RandomState(42)

    def train(self, timeout=None) -> None:
        for _ in range(self.n_partidas):
            trayectoria = self._jugar_partida()
            self._actualizar_q(trayectoria)

    def act(self, s: np.ndarray) -> int:
        state = ConnectState(s)
        acciones = state.get_free_cols()

        if not acciones:
            return None

        jugador = state.player

        for a in acciones:
            siguiente = state.transition(a)

            if siguiente.get_winner() == jugador:
                return a

        oponente = -jugador

        for a in acciones:
            siguiente = state.transition(a)

            if siguiente.get_winner() == oponente:
                return a

        vals = [self._q.get(s, a) for a in acciones]

        if max(vals) == 0:
            return int(self._rng.choice(acciones))

        return self._q.mejor_accion(s, acciones)

    def _elegir_accion(self, board: np.ndarray, acciones: list) -> int:
        if self._rng.rand() < self.epsilon:
            return int(self._rng.choice(acciones))

        return self._q.mejor_accion(board, acciones)

    def _jugar_partida(self) -> list:
        state = ConnectState(np.zeros((6, 7), dtype=int))
        trayectoria = []

        while not state.is_final():
            acciones = state.get_free_cols()

            accion = self._elegir_accion(state.board, acciones)

            trayectoria.append(
                (state.board.copy(), accion, state.player)
            )

            state = state.transition(accion)

        trayectoria.append((state.board.copy(), None, None))

        return trayectoria

    def _actualizar_q(self, trayectoria: list) -> None:
        board_terminal = trayectoria[-1][0]
        mi_jugador = trayectoria[0][2]

        state_terminal = ConnectState(board_terminal)
        ganador = state_terminal.get_winner()

        if ganador == mi_jugador:
            U = 1.0

        elif ganador == -mi_jugador:
            U = -1.0

        else:
            U = 0.0

        vistos = set()

        for board, accion, jugador in reversed(trayectoria[:-1]):

            if jugador != mi_jugador:
                U = -U

            par = (key(board), accion)

            if par not in vistos:
                vistos.add(par)
                self._q.update(board, accion, U)
