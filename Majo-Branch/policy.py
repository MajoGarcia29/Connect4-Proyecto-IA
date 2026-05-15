import numpy as np

from connect4.policy import Policy

from game_utils import (
    jugador_actual,
    acciones_legales,
    aplicar_accion,
    es_terminal,
    recompensa_final,
    hay_ganador,
    key
)

from qlearning import TablaQ


class FVMCPolicy(Policy):

    def __init__(
        self,
        n_partidas: int = 5000,
        epsilon: float = 0.1
    ):

        self.n_partidas = n_partidas

        self.epsilon = epsilon

        self._q = TablaQ()

        self._rng = np.random.RandomState(42)

    def mount(self, timeout=None) -> None:

        for _ in range(self.n_partidas):

            trayectoria = self._jugar_partida()

            self._actualizar_q(trayectoria)

    def act(
        self,
        s: np.ndarray
    ) -> int:

        acciones = acciones_legales(s)

        if not acciones:

            return None

        jugador = jugador_actual(s)

        
        for a in acciones:

            siguiente = aplicar_accion(s, a, jugador)

            if hay_ganador(siguiente, jugador):

                return a

       
        oponente = -jugador

        for a in acciones:

            siguiente = aplicar_accion(s, a, oponente)

            if hay_ganador(siguiente, oponente):

                return a

       
        vals = [
            self._q.get(s, a)
            for a in acciones
        ]

        if max(vals) == 0:

            return int(
                self._rng.choice(acciones)
            )

        return self._q.mejor_accion(
            s,
            acciones
        )

    def _elegir_accion(
        self,
        board: np.ndarray,
        acciones: list
    ) -> int:

        if self._rng.rand() < self.epsilon:

            return int(
                self._rng.choice(acciones)
            )

        return self._q.mejor_accion(
            board,
            acciones
        )

    def _jugar_partida(self) -> list:

        board = np.zeros(
            (6, 7),
            dtype=int
        )

        trayectoria = []

        while not es_terminal(board):

            acciones = acciones_legales(board)

            jugador = jugador_actual(board)

            accion = self._elegir_accion(
                board,
                acciones
            )

            trayectoria.append(
                (
                    board.copy(),
                    accion,
                    jugador
                )
            )

            board = aplicar_accion(
                board,
                accion,
                jugador
            )

        trayectoria.append(
            (
                board.copy(),
                None,
                None
            )
        )

        return trayectoria

    def _actualizar_q(
        self,
        trayectoria: list
    ) -> None:

        board_terminal = trayectoria[-1][0]

        mi_jugador = trayectoria[0][2]

        U = recompensa_final(
            board_terminal,
            mi_jugador
        )

        vistos = set()

        for board, accion, jugador in reversed(
            trayectoria[:-1]
        ):

            if jugador != mi_jugador:

                U = -U

            par = (
                key(board),
                accion
            )

            if par not in vistos:

                vistos.add(par)

                self._q.update(
                    board,
                    accion,
                    U
                )