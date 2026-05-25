"""
Entrenamiento de TBOPIPolicy mediante self-play y juego contra política externa.

Esquema:
  - El agente juega N partidas por ronda contra un pool de oponentes.
  - El pool contiene: versiones previas del agente (snapshots de la Q-table)
    y la política del compañero.
  - Al final de cada ronda se guarda un snapshot de la Q-table actual.
  - El agente siempre aprende (actualiza su Q-table) independientemente
    de contra quién juegue.

Uso:
    python train.py
    python train.py --rounds 50 --games_per_round 20 --opponent opponent/policy.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from connect4.connect_state import ConnectState

from policy import TBOPIPolicy


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    rounds: int            = 30       # rondas de entrenamiento
    games_per_round: int   = 20       # partidas por ronda
    best_of: int           = 1        # partidas por enfrentamiento dentro de ronda
    snapshot_every: int    = 5        # guardar snapshot cada N rondas
    max_pool_size: int     = 5        # máximo de snapshots en el pool de self-play
    opponent_path: str     = ""       # ruta al policy.py del compañero (vacío = solo self-play)
    qtable_path: str       = "qtable.json"
    snapshots_dir: str     = "snapshots"
    seed: int              = 42
    n_trials: int          = 200
    rollout_eps: float     = 0.3
    temperature: float     = 1.0
    first_player_prob: float = 0.5   # probabilidad de que el agente vaya primero


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_opponent_policy(path: str):
    """
    Carga dinámicamente una clase Policy desde un archivo externo.
    Asume que el archivo expone exactamente una subclase de Policy
    distinta de la base, o una clase llamada con 'Policy' en el nombre.
    """
    spec   = importlib.util.spec_from_file_location("opponent_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Buscar la primera clase que tenga 'Policy' en el nombre
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and "Policy" in name and name != "Policy":
            return obj
    raise ImportError(f"No se encontró ninguna clase Policy en {path}")


def snapshot_qtable(qtable_path: Path, snapshots_dir: Path, round_n: int) -> Path:
    """Copia la Q-table actual como snapshot numerado."""
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    dest = snapshots_dir / f"qtable_round_{round_n:04d}.json"
    shutil.copy(qtable_path, dest)
    return dest


def load_snapshot_policy(snapshot_path: Path, config: TrainConfig) -> TBOPIPolicy:
    """Crea un agente congelado que usa un snapshot de Q-table."""
    return TBOPIPolicy(
        n_trials    = config.n_trials,
        seed        = config.seed,
        rollout_eps = config.rollout_eps,
        temperature = config.temperature,
        qtable_path = str(snapshot_path),
        qtable_alpha= 0.0,   # alpha=0: no aprende, solo usa el snapshot
    )


# ---------------------------------------------------------------------------
# Jugar una partida
# ---------------------------------------------------------------------------

def play_game(
    agent: TBOPIPolicy,
    opponent,
    agent_goes_first: bool,
    rng: np.random.Generator,
) -> float:
    """
    Juega una partida completa entre agent y opponent.

    Retorna desde la perspectiva del agente:
      +1 victoria, -1 derrota, 0 empate.

    Convención del entorno:
      - player == -1 mueve primero
      - player == +1 mueve segundo
    """
    opp_instance = opponent()
    opp_instance.mount()
    agent.mount()

    # Asignar qué jugador numérico es el agente
    agent_player = -1 if agent_goes_first else 1

    state = ConnectState()

    while not state.is_final():
        if state.player == agent_player:
            action = agent.act(state.board)
        else:
            action = opp_instance.act(state.board)
        state = state.transition(int(action))

    winner = state.get_winner()
    if winner == agent_player:
        return 1.0
    if winner == 0:
        return 0.0
    return -1.0


# ---------------------------------------------------------------------------
# Una ronda de entrenamiento
# ---------------------------------------------------------------------------

@dataclass
class RoundStats:
    wins: int   = 0
    losses: int = 0
    draws: int  = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def winrate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0

    def record(self, result: float) -> None:
        if result > 0:
            self.wins += 1
        elif result < 0:
            self.losses += 1
        else:
            self.draws += 1

    def __str__(self) -> str:
        return (f"W={self.wins} L={self.losses} D={self.draws} "
                f"winrate={self.winrate:.1%}")


def run_round(
    agent: TBOPIPolicy,
    opponent_pool: list,          # lista de clases Policy (no instancias)
    games_per_round: int,
    first_player_prob: float,
    rng: np.random.Generator,
) -> RoundStats:
    """
    Juega `games_per_round` partidas sorteando oponente del pool en cada una.
    El agente aprende (actualiza Q-table) en cada partida.
    """
    stats = RoundStats()

    for _ in range(games_per_round):
        opponent       = opponent_pool[int(rng.integers(len(opponent_pool)))]
        agent_first    = rng.random() < first_player_prob
        result         = play_game(agent, opponent, agent_first, rng)
        stats.record(result)

    return stats


# ---------------------------------------------------------------------------
# Loop de entrenamiento principal
# ---------------------------------------------------------------------------

def train(config: TrainConfig) -> None:
    rng           = np.random.default_rng(config.seed)
    snapshots_dir = Path(config.snapshots_dir)
    qtable_path   = Path(config.qtable_path)

    # Agente principal: aprende durante todo el entrenamiento
    agent = TBOPIPolicy(
        n_trials     = config.n_trials,
        seed         = config.seed,
        rollout_eps  = config.rollout_eps,
        temperature  = config.temperature,
        qtable_path  = config.qtable_path,
        qtable_alpha = 0.1,
    )
    agent.mount()

    # Pool de oponentes: empieza con el propio agente (self-play puro)
    # Cada elemento es una clase/factory, no una instancia, porque
    # play_game llama opponent() para crear una instancia fresca por partida.
    snapshot_pool: list = []

    # Cargar política del compañero si se proveyó
    external_opponent = None
    if config.opponent_path:
        OpponentClass    = load_opponent_policy(config.opponent_path)
        external_opponent = OpponentClass
        print(f"Oponente externo cargado: {config.opponent_path}")

    # Historial de métricas
    history: list[dict] = []

    print(f"\n{'='*55}")
    print(f"  Iniciando entrenamiento: {config.rounds} rondas × "
          f"{config.games_per_round} partidas")
    print(f"{'='*55}\n")

    for round_n in range(1, config.rounds + 1):

        # Construir pool de oponentes para esta ronda
        # ─ Si no hay snapshots aún, el agente juega contra sí mismo
        #   instanciando con la Q-table actual (alpha=0 para no doble-aprender)
        if snapshot_pool:
            pool = snapshot_pool[:]
        else:
            # Usar al propio agente como oponente congelado temporalmente
            pool = [lambda: load_snapshot_policy(qtable_path, config)]

        # Añadir oponente externo al pool si existe
        if external_opponent is not None:
            pool.append(external_opponent)

        # Jugar la ronda
        stats = run_round(
            agent            = agent,
            opponent_pool    = pool,
            games_per_round  = config.games_per_round,
            first_player_prob= config.first_player_prob,
            rng              = rng,
        )

        # Registrar métricas
        entry = {
            "round":      round_n,
            "wins":       stats.wins,
            "losses":     stats.losses,
            "draws":      stats.draws,
            "winrate":    round(stats.winrate, 4),
            "pool_size":  len(pool),
        }
        history.append(entry)
        print(f"Ronda {round_n:3d}/{config.rounds}  |  {stats}  "
              f"| pool={len(pool)}")

        # Guardar snapshot periódicamente y agregarlo al pool
        if round_n % config.snapshot_every == 0:
            snap_path = snapshot_qtable(qtable_path, snapshots_dir, round_n)
            snapshot_pool.append(
                lambda p=snap_path: load_snapshot_policy(p, config)
            )
            # Mantener el pool acotado: descartar snapshots más viejos
            if len(snapshot_pool) > config.max_pool_size:
                snapshot_pool.pop(0)
            print(f"           → Snapshot guardado: {snap_path.name} "
                  f"(pool snapshots: {len(snapshot_pool)})")

    # Guardar historial de entrenamiento
    history_path = Path("train_history.json")
    history_path.write_text(json.dumps(history, indent=2))
    print(f"\nEntrenamiento finalizado. Historial: {history_path}")
    print(f"Q-table final: {qtable_path} "
          f"({len(agent._qtable)} entradas)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de TBOPIPolicy por self-play"
    )
    parser.add_argument("--rounds",            type=int,   default=30)
    parser.add_argument("--games_per_round",   type=int,   default=20)
    parser.add_argument("--snapshot_every",    type=int,   default=5)
    parser.add_argument("--max_pool_size",     type=int,   default=5)
    parser.add_argument("--n_trials",          type=int,   default=200)
    parser.add_argument("--rollout_eps",       type=float, default=0.3)
    parser.add_argument("--temperature",       type=float, default=1.0)
    parser.add_argument("--first_player_prob", type=float, default=0.5)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--qtable_path",       type=str,   default="qtable.json")
    parser.add_argument("--snapshots_dir",     type=str,   default="snapshots")
    parser.add_argument("--opponent",          type=str,   default="",
                        dest="opponent_path",
                        help="Ruta al policy.py del compañero")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
