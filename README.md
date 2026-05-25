# Agente FVMC para Connect-4

María José García  
Fundamentos de Inteligencia Artificial — 2026.1  
Universidad de La Sabana

---

# Descripción

Este proyecto implementa un agente inteligente para jugar Connect-4 utilizando **First-Visit Monte Carlo (FVMC)** con **Q-values**.

El agente aprende jugando partidas completas contra sí mismo (*self-play*) y almacena experiencia en una TablaQ, donde estima qué tan buena es cada acción en distintos estados del tablero.

Además del aprendizaje mediante Q-values, el agente incorpora dos heurísticas:

- Ganar inmediatamente si existe una jugada ganadora
- Bloquear al oponente si puede ganar en el siguiente turno

---

# Idea principal

El agente utiliza aprendizaje basado en experiencia acumulada.

Durante el entrenamiento:

1. Juega miles de partidas contra sí mismo
2. Guarda trayectorias de estados y acciones
3. Obtiene una recompensa final:
   - Victoria → +1
   - Derrota → -1
   - Empate → 0
4. Actualiza los Q-values usando promedio incremental

La política de entrenamiento utiliza exploración epsilon-greedy.

---

# Estructura de archivos

```text
mi_agente/
│
├── policy.py
├── guardar_qvalues.py
├── qvalues_majo.pkl
├── entrega.ipynb
├── README.md
```

- `policy.py` → implementación principal del agente FVMC
- `guardar_qvalues.py` → script para entrenar el agente y guardar los Q-values
- `qvalues_majo.pkl` → archivo con los Q-values entrenados
- `entrega.ipynb` → notebook con experimentos, análisis y gráficas
- `README.md` → guía de uso

---

# Requisitos

Python 3.10+

Librerías necesarias:

```bash
pip install numpy matplotlib
```

---

# Generar Q-values

Antes de ejecutar el notebook `entrega.ipynb`, es necesario generar el archivo:

```text
qvalues_majo.pkl
```

Este archivo se crea ejecutando:

```bash
python guardar_qvalues.py
```

El notebook utiliza este archivo `.pkl` para cargar los Q-values previamente entrenados y realizar las evaluaciones y gráficas.

---

# Uso del agente

## Entrenar el agente

```python
from policy import FVMCPolicy

agente = FVMCPolicy(
    n_partidas=5000,
    epsilon=0.1
)

agente.train()
```

---

## Obtener una acción

```python
accion = agente.act(board)
```

donde `board` es un `numpy.ndarray` de tamaño `(6,7)`.

---

# Parámetros importantes

| Parámetro | Descripción |
|---|---|
| `n_partidas` | Cantidad de partidas de entrenamiento |
| `epsilon` | Probabilidad de exploración aleatoria |

---

# Resultados

El agente:

- Supera consistentemente al jugador aleatorio
- Mantiene win rate >90%
- Funciona jugando como primer y segundo jugador
- Mejora progresivamente con más entrenamiento

---

# Posibles mejoras

- Epsilon adaptativo
- Más entrenamiento
- Entrenamiento contra agentes más fuertes
- Generalización mediante redes neuronales

---

# Autor

María José García
