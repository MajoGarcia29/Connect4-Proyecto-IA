import numpy as np


def jugador_actual(board: np.ndarray) -> int:
    return -1 if np.sum(board == -1) == np.sum(board == 1) else 1 #Como asi que else 1?
#Determina de quien es el turno
#El rojo(-1) empieza siempre
#np.sum(board == -1) cuenta cuantas fichas rojas hay
#np.sum(board == 1) cuenta cuantas fichas amarillas hay

def acciones_legales(board: np.ndarray) -> list:
    return [c for c in range(7) if board[0, c] == 0]
#Devuelve una lista con las clumnas donde todavia se puede colocar una ficha


def aplicar_accion(board: np.ndarray, col: int, player: int) -> np.ndarray:

    nuevo = board.copy()
    for fila in reversed(range(6)): #  Recorre las filas de abajo hacia arriba porque las fichas caen 
        if nuevo[fila, col] == 0:
            nuevo[fila, col] = player
            break

    return nuevo
#Aplica la accion de colocar una ficha del jugador en la columna especificada


def hay_ganador(board: np.ndarray, player: int) -> bool: #determina si el jugador seleccionado hizo un 4 en fila revisando 4 direcciones

    for r in range(6):
        for c in range(7):

            if board[r, c] != player:
                continue

            if (c + 3 < 7 and all( board[r, c+i] == player
                    for i in range(4)
                )
            ):
                return True

            if (
                r + 3 < 6
                and all(
                    board[r+i, c] == player
                    for i in range(4)
                )
            ):
                return True

            if (
                r + 3 < 6
                and c + 3 < 7
                and all(
                    board[r+i, c+i] == player
                    for i in range(4)
                )
            ):
                return True

            if (
                r + 3 < 6
                and c - 3 >= 0
                and all(
                    board[r+i, c-i] == player
                    for i in range(4)
                )
            ):
                return True

    return False


def es_terminal(board: np.ndarray) -> bool:

    return (
        hay_ganador(board, -1)
        or hay_ganador(board, 1)
        or len(acciones_legales(board)) == 0
    )


def recompensa_final(board: np.ndarray, mi_jugador: int) -> float:

    if hay_ganador(board, mi_jugador):
        return 1.0

    if hay_ganador(board, -mi_jugador):
        return -1.0

    return 0.0


def key(board: np.ndarray) -> bytes:
    return board.tobytes()