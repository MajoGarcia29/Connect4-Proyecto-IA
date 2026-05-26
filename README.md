# TBOPIPolicy — Connect 4

Guía técnica para la ejecución y entrenamiento del agente.

---

## Estructura de archivos

```
policy.py          # Agente TBOPIPolicy listo para el torneo
train.py           # Script de entrenamiento por self-play
qtable.json        # Conocimiento acumulado (se crea al entrenar)
snapshots/         # Versiones anteriores del agente (se crean al entrenar)
train_history.json # Métricas de entrenamiento por ronda
```

---

## Requisitos

```
numpy
connect4   # paquete del entorno (connect4.policy, connect4.connect_state)
```


## Cómo usar `tbopiv3` (Versión Avanzada)

### Uso mínimo en el torneo

```python
from tbopiv3.policy import TBOPIPolicy

policy = TBOPIPolicy()
policy.mount()
action = policy.act(board)  # board es un np.ndarray del entorno
```

Esta versión utiliza una Q-table para persistir el conocimiento y LGRF para aprender respuestas tácticas durante el turno.

### Usar una Q-table entrenada

```python
policy = TBOPIPolicy(qtable_path="tbopiv3/qtable.pkl")
policy.mount()  # carga la Q-table desde disco automáticamente
```

`mount()` debe llamarse antes de `act()`. En el torneo el entorno lo llama
automáticamente, pero si usas la política de forma manual asegúrate de
llamarlo.

### Parámetros configurables

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `n_trials` | `int` | `200` | Simulaciones Monte Carlo por turno. Más trials = mejor decisión pero más lento. |
| `seed` | `int` | `42` | Semilla aleatoria. Fija para reproducibilidad, varía para diversidad. |
| `ucb_c` | `float` | `√2` | Constante de exploración UCB. Valores altos exploran más; valores bajos explotan más. |
| `rollout_eps` | `float` | `0.3` | Probabilidad de jugada aleatoria en rollout. `0.0` = siempre sigue LGRF/Q-table; `1.0` = siempre aleatorio. |
| `qtable_path` | `str` | `"qtable.pkl"` | Ruta de la Q-table persistente. |
| `qtable_alpha` | `float` | `0.1` | Tasa de aprendizaje TD(0). `0.0` = no aprende (modo congelado). |

### Ejemplo con parámetros personalizados

```python
policy = TBOPIPolicy(
    n_trials      = 400,        # más simulaciones por turno
    rollout_eps   = 0.2,        # menos aleatoriedad en rollouts
    qtable_path   = "qtable.pkl",
    qtable_alpha  = 0.05,       # aprendizaje más conservador
)
```

### Modo congelado (solo inferencia, sin aprendizaje)

```python
policy = TBOPIPolicy(
    qtable_path  = "qtable.pkl",
    qtable_alpha = 0.0,   # alpha=0: no modifica la Q-table
)
```

Útil para participar en el torneo final sin que el agente siga modificando
su Q-table durante las partidas.

---

## Cómo usar `tbopiv2` (Versión Ligera)

La versión 2 es una implementación autocontenida que no utiliza memoria persistente. Basa su fuerza en la velocidad de las simulaciones Monte Carlo y una heurística de proximidad al centro.

### Uso mínimo

```python
from tbopiv2.policy import TBOPIPolicy

policy = TBOPIPolicy(n_trials=300)
policy.mount()
action = policy.act(board)
```

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `n_trials` | `int` | `200` | Simulaciones por turno. |
| `seed` | `int` | `42` | Semilla aleatoria para reproducibilidad. |
| `ucb_c` | `float` | `1.414` | Constante de exploración UCB. |
| `rollout_eps`| `float` | `0.3` | Aleatoriedad en las simulaciones. |

---

### Comandos principales

| Tarea | Comando |
|---|---|
| Iniciar entrenamiento base | `python train.py` |
| Entrenar contra oponente | `python train.py --opponent ruta/policy.py` |
| Continuar desde Q-table | `python train.py --qtable_path qtable.pkl` |
| Ajuste de parámetros | `python train.py --rounds 50 --games_per_round 30 --n_trials 300` |

### Flujo recomendado

1. **Fase Base**: `python train.py --rounds 30 --games_per_round 20`
2. **Fase de Ajuste**: `python train.py --rounds 20 --games_per_round 30 --opponent rival/policy.py`
3. **Finalización**: Usar `qtable_alpha=0.0` en la configuración de `TBOPIPolicy` para competir.
