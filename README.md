# TBOPIPolicy — Connect 4

Implementación de un agente de Connect 4 basado en **Trial-Based Online Policy
Improvement (TBOPI)** extendido con tres capas de aprendizaje:

- **UCB** para selección inteligente de trials dentro del turno
- **LGRF** (Last Good Reply with Forgetting) para aprender respuestas durante
  los rollouts del turno actual
- **Q-table persistente por features** para acumular conocimiento estratégico
  entre partidas

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

---

## Cómo usar `policy.py`

### Uso mínimo en el torneo

```python
from policy import TBOPIPolicy

policy = TBOPIPolicy()
policy.mount()
action = policy.act(board)  # board es un np.ndarray del entorno
```

La política es compatible con la interfaz `Policy` del entorno y funciona
sin entrenamiento previo. Sin `qtable.json` en disco, el agente comienza
con conocimiento vacío y aprende desde cero en cada turno.

### Usar una Q-table entrenada

```python
policy = TBOPIPolicy(qtable_path="qtable.json")
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
| `temperature` | `float` | `1.0` | Temperatura del softmax final (PImp). `→ 0` = greedy puro; `→ ∞` = uniforme. |
| `qtable_path` | `str` | `"qtable.json"` | Ruta de la Q-table persistente. |
| `qtable_alpha` | `float` | `0.1` | Tasa de aprendizaje TD(0). `0.0` = no aprende (modo congelado). |

### Ejemplo con parámetros personalizados

```python
policy = TBOPIPolicy(
    n_trials      = 400,        # más simulaciones por turno
    rollout_eps   = 0.2,        # menos aleatoriedad en rollouts
    temperature   = 0.5,        # selección final más greedy
    qtable_path   = "qtable.json",
    qtable_alpha  = 0.05,       # aprendizaje más conservador
)
```

### Modo congelado (solo inferencia, sin aprendizaje)

```python
policy = TBOPIPolicy(
    qtable_path  = "qtable.json",
    qtable_alpha = 0.0,   # alpha=0: no modifica la Q-table
)
```

Útil para participar en el torneo final sin que el agente siga modificando
su Q-table durante las partidas.

---

## Cómo usar `train.py`

### Entrenamiento básico (solo self-play)

```bash
python train.py
```

El agente juega 30 rondas × 20 partidas contra versiones anteriores de sí
mismo. La Q-table se guarda en `qtable.json` y los snapshots en `snapshots/`.

### Entrenamiento contra el compañero

```bash
python train.py --opponent ruta/al/compañero/policy.py
```

El oponente externo se añade al pool de self-play. En cada partida se
sortea aleatoriamente entre los snapshots del agente y el compañero,
así el agente aprende contra ambos estilos simultáneamente.

### Continuar un entrenamiento previo

```bash
python train.py --qtable_path qtable.json
```

Si `qtable.json` ya existe, el agente retoma desde el conocimiento acumulado.
El entrenamiento es acumulativo: se puede interrumpir y continuar en cualquier
momento.

### Todos los parámetros CLI

```bash
python train.py \
  --rounds 50 \              # número de rondas
  --games_per_round 30 \     # partidas por ronda
  --snapshot_every 5 \       # guardar snapshot cada N rondas
  --max_pool_size 5 \        # máximo de snapshots en el pool
  --n_trials 300 \           # trials por turno del agente
  --rollout_eps 0.25 \       # épsilon del rollout
  --temperature 0.8 \        # temperatura PImp
  --first_player_prob 0.5 \  # prob. de que el agente vaya primero
  --seed 42 \
  --qtable_path qtable.json \
  --snapshots_dir snapshots \
  --opponent compañero/policy.py
```

### Salida del entrenamiento

Durante la ejecución se imprime una línea por ronda:

```
Ronda   5/30  |  W=14 L=4 D=2  winrate=70.0%  | pool=1
           → Snapshot guardado: qtable_round_0005.json (pool snapshots: 1)
Ronda  10/30  |  W=12 L=6 D=2  winrate=60.0%  | pool=2
           → Snapshot guardado: qtable_round_0010.json (pool snapshots: 2)
```

Al terminar se genera `train_history.json` con el historial completo para
graficar la curva de aprendizaje:

```json
[
  {"round": 1, "wins": 12, "losses": 7, "draws": 1, "winrate": 0.6, "pool_size": 1},
  {"round": 2, "wins": 13, "losses": 6, "draws": 1, "winrate": 0.65, "pool_size": 1},
  ...
]
```

### Flujo de entrenamiento recomendado

```bash
# 1. Entrenamiento inicial de base (solo self-play)
python train.py --rounds 30 --games_per_round 20

# 2. Afinar contra el compañero
python train.py --rounds 20 --games_per_round 30 --opponent compañero/policy.py

# 3. Para el torneo: congelar la Q-table (qtable_alpha=0 en policy.py)
#    o simplemente no entrenar más y entregar policy.py + qtable.json
```

---

## Explicación conceptual completa

### El problema central

Connect 4 tiene aproximadamente 4 billones de posiciones posibles. Un agente
no puede analizarlas todas, así que necesita una estrategia para tomar
decisiones buenas en tiempo limitado. La solución de TBOPI es simular partidas
aleatorias y usar sus resultados para estimar cuán buena es cada columna.

---

### `policy.py` — capa por capa

#### Lógica de juego: `get_free_cols`, `apply_move`, `check_winner`, `current_player`

Son funciones puras sin estado: dado un tablero, devuelven información sobre
él sin modificarlo. `apply_move` siempre devuelve una **copia nueva** del
tablero, lo cual es fundamental para que los rollouts no corrompan el estado
real de la partida.

`current_player` infiere el turno contando fichas: si hay igual número de
fichas `+1` y `-1`, le toca a `-1` (el primer jugador por convención). Esto
evita tener que pasar el turno como parámetro adicional.

#### Features del tablero: `board_features` y `_count_open_threats`

En vez de recordar tableros exactos (imposible por el tamaño del espacio),
el agente resume cada posición en tres números:

- `own_threats`: cuántas secuencias de 3 fichas propias tienen al menos un
  extremo libre (pueden completarse en cuatro).
- `opp_threats`: lo mismo para el oponente.
- `center_ctrl`: cuántas fichas propias hay en las tres columnas centrales.

Dos tableros completamente distintos pero con la misma estructura estratégica
(mismas amenazas, mismo control del centro) producen el mismo vector de
features y **comparten el conocimiento aprendido**. Eso es generalización:
el agente aprende de situaciones similares aunque nunca haya visto ese
tablero exacto.

#### Q-table persistente: `load_qtable`, `save_qtable`, `_update_qtable`

La Q-table es un diccionario que asocia `(player, own_threats, opp_threats,
center_ctrl, columna)` con un valor numérico que representa qué tan buena
es esa columna en ese contexto estratégico.

La actualización usa la regla **TD(0)**:

```
Q(s, a) ← Q(s, a) + α × (reward - Q(s, a))
```

Que es equivalente a calcular una media ponderada exponencialmente. Con
`α = 0.1`, cada nuevo reward mueve el valor un 10% hacia él, lo que hace
que el valor converja lentamente pero de forma estable frente al ruido de
los rollouts individuales. La Q-table se serializa como JSON para persistir
entre ejecuciones.

#### Tabla LGRF: `_update_lgrf`

LGRF (Baier & Winands, 2014) es una memoria intra-turno. Empieza vacía al
inicio de cada llamada a `act()` y se llena durante los trials de ese turno.

La tabla asocia `(player, columna_del_oponente)` con `columna_respuesta`.
Después de cada rollout, si el resultado fue victoria, se registra cada par
`(jugada_del_oponente → mi_respuesta)` a lo largo de toda la secuencia
ganadora. Si fue derrota, esas entradas se eliminan. Los empates no modifican
la tabla.

La diferencia clave con la Q-table: LGRF aprende **dentro del presupuesto
del turno actual**. Los últimos trials del presupuesto son mejores que los
primeros porque la tabla ya tiene información de los trials anteriores. MCTS
estándar no hace esto.

#### Política de rollout: `rollout_policy`

Tres niveles de prioridad decreciente para elegir la columna en cada paso
del rollout:

**Nivel 1 — LGRF.** Si hay una respuesta aprendida para la última jugada del
oponente y esa columna sigue libre, se usa directamente. Es el conocimiento
más específico y reciente: fue aprendido en este turno, en esta partida.

**Nivel 2 — Q-table.** Con probabilidad `1 - eps`, se consulta la Q-table
para el contexto de features actual y se elige la columna con mayor Q-value
aprendido entre partidas. Es conocimiento generalizado que reemplaza por
completo la heurística estática de columna central: en vez de preferir
siempre el centro, el agente prefiere lo que históricamente ha funcionado
en situaciones estratégicamente similares.

**Nivel 3 — Aleatorio.** Con probabilidad `eps`, jugada uniforme entre las
columnas libres. Garantiza exploración y evita que el agente quede atrapado
en patrones locales.

#### Motor de simulaciones: `run_trial`

Ejecuta una partida completa desde un tablero dado hasta que alguien gana o
hay empate. Registra el historial `(jugador, columna)` de cada jugada, y al
terminar actualiza la tabla LGRF con el resultado.

Ambos jugadores usan la misma política de rollout. Esta simplificación
funciona porque con muchos trials el promedio de recompensas converge al
valor real de cada acción.

#### Selección UCB: `_ucb_select`

Antes de cada rollout, UCB decide qué columna simular:

```
UCB(a) = Q(a) + c × √(ln(N) / n(a))
```

El primer término explota lo que ya se sabe (columnas con Q alto).
El segundo término explora lo que poco se ha visto (columnas con pocas
visitas). `c = √2` balancea ambos objetivos de forma óptima en teoría
(Kocsis & Szepesvári, 2006). Las columnas no visitadas reciben prioridad
absoluta (UCB infinito).

#### El método `act`: el algoritmo completo

Cuando el entorno llama a `act(board)` ocurren tres fases en orden:

**Fase 1 — Respuesta inmediata.** Antes de gastar trials, revisa si hay una
victoria disponible o un bloqueo necesario. Sin esto el agente podría perder
partidas ganadas si los trials no convergen a tiempo.

**Fase 2 — Loop de trials.** Inicializa `q_local` y `n_local` en cero, y
una tabla LGRF vacía. Por cada trial: selecciona columna con UCB, hace un
rollout completo con la política LGRF → Q-table → ε-greedy, actualiza
`q_local[a]` con la media incremental de Welford, y actualiza la Q-table
persistente con el reward observado. Al terminar todos los trials, guarda
la Q-table en disco.

**Fase 3 — Paso PImp.** Convierte los `q_local` en una distribución de
probabilidad con softmax y temperatura, y samplea la acción final. La
diferencia con MCTS es que en vez de `argmax(Q)` mantiene incertidumbre:
si dos columnas tienen Q similar, ambas tienen probabilidad apreciable.

---

### `train.py` — capa por capa

#### `TrainConfig`

Dataclass que agrupa todos los hiperparámetros del entrenamiento. Al usar
el CLI, `argparse` parsea los argumentos y construye esta estructura, lo
que mantiene la función `train()` desacoplada de cómo se invoca.

#### `load_opponent_policy`

Carga dinámicamente un archivo `policy.py` externo sin necesidad de
instalarlo como paquete. Usa `importlib` para ejecutar el módulo en tiempo
de ejecución y busca la primera clase cuyo nombre contenga `"Policy"`.
Esto permite cargar la política del compañero sin modificar el código.

#### `snapshot_qtable` y `load_snapshot_policy`

Un snapshot es una copia congelada de la Q-table en un punto del
entrenamiento. `load_snapshot_policy` crea un agente que usa ese snapshot
con `qtable_alpha=0.0`, es decir, no aprende: solo usa el conocimiento
guardado en ese momento. Esto garantiza que los oponentes del pool sean
referencias estables y no estén aprendiendo al mismo tiempo que el agente
principal (lo que crearía un loop inestable).

#### `play_game`

Ejecuta una partida completa entre el agente y un oponente. Instancia el
oponente fresco por partida (llamando `opponent()`) para evitar que el
estado interno de una partida contamine la siguiente. El resultado se
devuelve desde la perspectiva del agente: `+1`, `-1` o `0`.

#### `RoundStats`

Dataclass simple que acumula victorias, derrotas y empates de una ronda y
calcula el winrate. Separar las estadísticas en su propia clase hace que
el código de `run_round` sea más legible y que el historial sea fácil de
serializar a JSON.

#### `run_round`

Juega `games_per_round` partidas sorteando un oponente aleatorio del pool
en cada una. El sorteo uniforme entre todos los oponentes del pool garantiza
que el agente se exponga equitativamente a todos los estilos: versiones
anteriores de sí mismo y el oponente externo.

#### `train` — el loop principal

Esta es la función central. Su lógica es:

**Inicialización.** Crea el agente principal con `qtable_alpha=0.1` (aprende)
y carga su Q-table desde disco si existe. Si hay un oponente externo, lo
carga con `load_opponent_policy`.

**Por cada ronda.** Construye el pool de oponentes combinando los snapshots
disponibles y el oponente externo. Si aún no hay snapshots (primeras rondas),
el agente juega contra una copia congelada de sí mismo. Ejecuta `run_round`,
registra las estadísticas, y cada `snapshot_every` rondas guarda un snapshot
y lo agrega al pool.

**Pool sliding window.** El pool mantiene como máximo `max_pool_size`
snapshots, descartando los más antiguos. Esto imita la estrategia de
AlphaGo Zero: jugar contra versiones recientes pero no demasiado antiguas,
evitando que el agente optimice contra una versión de sí mismo que ya
quedó obsoleta.

**Finalización.** Guarda `train_history.json` con las métricas por ronda
y reporta el tamaño final de la Q-table.

---

## Capas de conocimiento y su ciclo de vida

| Capa | Alcance | Persiste | Propósito |
|---|---|---|---|
| `q_local` | Un turno | No | Estimación de valor por columna en este turno |
| LGRF | Un turno | No | Respuestas aprendidas dentro del presupuesto de trials |
| Q-table | Entre partidas | Sí (`qtable.json`) | Conocimiento estratégico generalizado por features |

El diseño en tres capas garantiza que cada tipo de conocimiento opere en
su escala temporal apropiada: lo más específico y volátil (LGRF) vive solo
un turno; lo más general y duradero (Q-table) persiste indefinidamente.

---

## Referencias

- Kocsis & Szepesvári (2006). *Bandit based Monte-Carlo planning.* ECML.
- Coulom (2006). *Efficient selectivity and backup operators in Monte-Carlo tree search.* CG.
- Rosin (2011). *Multi-armed bandits with episode context.* AAMAS.
- Baier & Winands (2014). *MCTS-Minimax Hybrids.* IEEE TCIAIG.
- Silver et al. (2016). *Mastering the game of Go with deep neural networks and tree search.* Nature.
