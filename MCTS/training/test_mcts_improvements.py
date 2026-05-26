"""
Test Comparativo: MCTS sin Q-values vs MCTS con Q-values Pre-entrenados
========================================================================

Este script:
  1. Entrena un agente MCTS base jugando 50 partidas contra un aleatorio
  2. Acumula Q-values del árbol MCTS tras cada partida
  3. Guarda los Q-values en archivo pickle
  4. Los carga en un nuevo agente MCTS
  5. Compara ambos agentes (con y sin Q-values) contra jugador aleatorio
  6. Muestra resultados con tasa de victoria y mejora porcentual
"""

import sys
import os
import time
import random
import importlib.util
import numpy as np

# ── Configurar path del proyecto ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

# ── Importar MCTS desde "Group A" (requiere importlib por espacio en nombre) ──
_mcts_path = os.path.join(_script_dir, "groups", "Group A", "MCTS.py")
spec = importlib.util.spec_from_file_location("MCTS", _mcts_path)
mcts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcts)

from connect4.connect_state import ConnectState

# Importar componentes del agente mejorado
MCTSAgent = mcts.MCTSAgent
extract_q_values = mcts.extract_q_values
merge_q_values = mcts.merge_q_values
save_q_values = mcts.save_q_values
load_q_values = mcts.load_q_values

# ── Configuración ──
TRAIN_GAMES = 50       # Partidas de entrenamiento para generar Q-values
TEST_GAMES = 100       # Partidas de testing para cada agente
TRAIN_ITERATIONS = 1000  # Iteraciones MCTS durante entrenamiento
TEST_ITERATIONS = 1000   # Iteraciones MCTS durante testing
Q_VALUES_PATH = os.path.join(_script_dir, "q_values_trained.pkl")


# ============================================================
# Agente Aleatorio (oponente de referencia)
# ============================================================

class RandomAgent:
    """Jugador que elige columnas libres al azar."""

    def act(self, board):
        free_cols = [c for c in range(7) if board[0, c] == 0]
        return random.choice(free_cols)

    def mount(self, *args, **kwargs):
        pass


# ============================================================
# Funciones de Juego
# ============================================================

def play_game(agent1, agent2):
    """Juega una partida completa. agent1 juega como -1, agent2 como 1.
    
    Returns:
        Ganador (-1, 1) o 0 (empate).
    """
    state = ConnectState()
    while not state.is_final():
        if state.player == -1:
            action = agent1.act(state.board)
        else:
            action = agent2.act(state.board)
        state = state.transition(int(action))
    return state.get_winner()


def play_match(agent1, agent2, num_games):
    """Juega múltiples partidas alternando quién va primero.
    
    En partidas pares, agent1 juega primero (-1).
    En partidas impares, agent2 juega primero (-1).
    
    Returns:
        (victorias_agent1, victorias_agent2, empates)
    """
    wins1, wins2, draws = 0, 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            # agent1 juega como -1 (primero)
            winner = play_game(agent1, agent2)
            if winner == -1:
                wins1 += 1
            elif winner == 1:
                wins2 += 1
            else:
                draws += 1
        else:
            # agent2 juega como -1 (primero), agent1 como 1
            winner = play_game(agent2, agent1)
            if winner == 1:
                wins1 += 1
            elif winner == -1:
                wins2 += 1
            else:
                draws += 1
    return wins1, wins2, draws


# ============================================================
# Fase 1: Entrenamiento — Acumular Q-values
# ============================================================

def train_and_collect_q_values():
    """Entrena un agente base contra aleatorio y acumula Q-values."""
    print("=" * 65)
    print("  FASE 1: Entrenamiento del agente base")
    print(f"  {TRAIN_GAMES} partidas | {TRAIN_ITERATIONS} iteraciones/turno")
    print("=" * 65)

    # Agente sin Q-values pre-entrenados
    train_agent = MCTSAgent(iterations=TRAIN_ITERATIONS, pretrained_q_values={})
    random_opp = RandomAgent()
    accumulated_q = {}

    wins, losses, draws = 0, 0, 0
    start_time = time.time()

    for i in range(TRAIN_GAMES):
        # Alternar quién va primero para diversidad de estados
        if i % 2 == 0:
            winner = play_game(train_agent, random_opp)
            if winner == -1:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
        else:
            winner = play_game(random_opp, train_agent)
            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                draws += 1

        # Extraer Q-values del árbol de esta partida
        if train_agent.last_root is not None:
            game_q = extract_q_values(train_agent.last_root)
            accumulated_q = merge_q_values(accumulated_q, game_q)

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (TRAIN_GAMES - i - 1)
        print(
            f"  Partida {i + 1:3d}/{TRAIN_GAMES} | "
            f"W:{wins} L:{losses} D:{draws} | "
            f"Estados: {len(accumulated_q):,} | "
            f"ETA: {remaining:.0f}s",
            end="\r"
        )

    elapsed = time.time() - start_time
    print(f"\n\n  ✓ Entrenamiento completado en {elapsed:.1f}s")
    print(f"  ✓ Q-values acumulados: {len(accumulated_q):,} estados únicos")
    print(f"  ✓ Resultado entrenamiento: {wins}W / {losses}L / {draws}D")

    # Guardar Q-values
    save_q_values(accumulated_q, Q_VALUES_PATH)
    file_size = os.path.getsize(Q_VALUES_PATH) / 1024
    print(f"  ✓ Q-values guardados en: {Q_VALUES_PATH} ({file_size:.1f} KB)")

    return accumulated_q


# ============================================================
# Fase 2: Testing Comparativo
# ============================================================

def test_agent(agent_name, agent, num_games):
    """Testa un agente contra aleatorio y muestra progreso."""
    print(f"\n  Testeando: {agent_name} ({num_games} partidas)...")
    random_opp = RandomAgent()
    start_time = time.time()

    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(agent, random_opp)
            if winner == -1:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
        else:
            winner = play_game(random_opp, agent)
            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                draws += 1

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (num_games - i - 1)
        print(
            f"    Partida {i + 1:3d}/{num_games} | "
            f"W:{wins} L:{losses} D:{draws} | "
            f"ETA: {remaining:.0f}s",
            end="\r"
        )

    elapsed = time.time() - start_time
    win_rate = wins / num_games * 100
    print(f"\n    ✓ Completado en {elapsed:.1f}s | Tasa de victoria: {win_rate:.1f}%")

    return wins, losses, draws, elapsed


def run_comparison():
    """Compara agentes con y sin Q-values pre-entrenados."""
    print("\n" + "=" * 65)
    print("  FASE 2: Testing Comparativo")
    print(f"  {TEST_GAMES} partidas por agente | {TEST_ITERATIONS} iteraciones/turno")
    print("=" * 65)

    # Cargar Q-values entrenados
    loaded_q = load_q_values(Q_VALUES_PATH)
    if loaded_q is None:
        print("  ✗ Error: No se encontraron Q-values entrenados.")
        return

    print(f"  ✓ Q-values cargados: {len(loaded_q):,} estados")

    # Crear agentes
    # Agente base: sin Q-values, pero con smart rollout (center priority + win/block)
    agent_base = MCTSAgent(
        iterations=TEST_ITERATIONS, pretrained_q_values={}
    )
    # Agente mejorado: con Q-values pre-entrenados
    agent_with_q = MCTSAgent(
        iterations=TEST_ITERATIONS, pretrained_q_values=loaded_q
    )

    # Testear ambos
    w1, l1, d1, t1 = test_agent("MCTS Base (sin Q-values)", agent_base, TEST_GAMES)
    w2, l2, d2, t2 = test_agent("MCTS + Q-values Pre-entrenados", agent_with_q, TEST_GAMES)

    # ── Resultados ──
    rate1 = w1 / TEST_GAMES * 100
    rate2 = w2 / TEST_GAMES * 100
    improvement = rate2 - rate1

    print("\n" + "=" * 65)
    print("  RESULTADOS FINALES")
    print("=" * 65)
    print(f"  {'Métrica':<32} {'Sin Q-vals':>12} {'Con Q-vals':>12}")
    print("  " + "-" * 56)
    print(f"  {'Victorias':<32} {w1:>12} {w2:>12}")
    print(f"  {'Derrotas':<32} {l1:>12} {l2:>12}")
    print(f"  {'Empates':<32} {d1:>12} {d2:>12}")
    print(f"  {'Tasa de victoria':<32} {rate1:>11.1f}% {rate2:>11.1f}%")
    print(f"  {'Tiempo total':<32} {t1:>10.1f}s {t2:>10.1f}s")
    print(f"  {'Tiempo por partida':<32} {t1/TEST_GAMES:>10.2f}s {t2/TEST_GAMES:>10.2f}s")
    print("  " + "-" * 56)

    if improvement > 0:
        print(f"\n  ✓ Mejora porcentual con Q-values: +{improvement:.1f}%")
    elif improvement == 0:
        print(f"\n  → Sin cambio en tasa de victoria ({rate1:.1f}%)")
    else:
        print(f"\n  ✗ Reducción con Q-values: {improvement:.1f}%")
        print("    (Puede mejorar con más partidas de entrenamiento)")

    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Fix encoding para consola Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print()
    print("  +===========================================================+")
    print("  |     MCTS Connect-4: Test de Mejoras con Q-values          |")
    print("  +===========================================================+")
    print()

    total_start = time.time()

    # Fase 1: Entrenar y acumular Q-values
    q_values = train_and_collect_q_values()

    # Fase 2: Comparar agentes
    run_comparison()

    total_elapsed = time.time() - total_start
    print(f"  Tiempo total de ejecución: {total_elapsed:.1f}s")
    print()
