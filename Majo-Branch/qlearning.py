from game_utils import key


class TablaQ:

    def __init__(self):
        self._q = {}
        self._n = {}

    def get(self, board, accion) -> float:

        return self._q.get(
            key(board),
            {}
        ).get(
            accion,
            0.0
        )

    def update(self, board, accion, U: float) -> None:

        k = key(board)

        if k not in self._q:
            self._q[k] = {}
            self._n[k] = {}

        if accion not in self._q[k]:
            self._q[k][accion] = 0.0
            self._n[k][accion] = 0
        self._n[k][accion] += 1
        n = self._n[k][accion]
        self._q[k][accion] += (
            U - self._q[k][accion]
        ) / n

    def mejor_accion(
        self,
        board,
        acciones: list
    ) -> int:

        vals = self._q.get(
            key(board),
            {}
        )

        return max(
            acciones,
            key=lambda a: vals.get(a, 0.0)
        )