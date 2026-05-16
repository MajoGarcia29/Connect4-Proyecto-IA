from game_utils import key


class TablaQ:# Esta clase es básicamente una tabla de memoria del agente. Guarda qué tan buena fue cada jugada en cada situación.

    def __init__(self):
        self._q = {}
        self._n = {}
#Crea una tabla Q con dos diccionarios, uno para los valores Q y otro para cuantas veces se probo esa accion
    def get(self, board, accion) -> float:
        return self._q.get(key(board), {}).get(accion, 0.0)
#¿Qué tan bueno fue jugar columna 3 en este tablero?→ si no lo sabe, devuelve 0.0
    def update(self, board, accion, U: float) -> None:
#U es:la utilidad observada
        k = key(board) #Convierte tablero en clave.

        if k not in self._q:#si no ha visto este tablero, crea un espacio para guardarlo
            self._q[k] = {}
            self._n[k] = {}

        if accion not in self._q[k]:#si no ha probado la accion, inicializa su valor y contador
            self._q[k][accion] = 0.0
            self._n[k][accion] = 0

        self._n[k][accion] += 1 #contador para cuántas veces vio esa acción.
        n = self._n[k][accion]
        self._q[k][accion] += ( U - self._q[k][accion]) / n #Esto actualiza el promedio

    def mejor_accion(self, board, acciones: list) -> int:

        vals = self._q.get(key(board), {}) #Busca valores del estado

        return max(
            acciones,
            key=lambda a: vals.get(a, 0.0)
        ) #elige:la acción con mayor Q-value.