# Agente MCTS + Q-Values para Connect-4
**Inteligencia Artificial — 2026**

## Descripción
Este proyecto implementa un agente inteligente para jugar Connect-4 utilizando **Monte Carlo Tree Search (MCTS)** reforzado con **Q-values pre-entrenados**.

El agente basa su proceso de toma de decisiones en simulaciones de MCTS, pero acelera y mejora su exploración al inicializar los nodos no visitados con experiencia acumulada (Q-values).

## Estructura de archivos
```
Group /
│
├── MCTS.py
├── q_values.pkl
├── README.md
│
(Raíz del torneo)
└── test_mcts_improvements.py
```
- `MCTS.py` → Implementación principal del agente MCTSAgent y la política de juego.
- `q_values.pkl` → Archivo con los Q-values entrenados (memoria del agente).
- `README.md` → Esta guía de uso.
- `test_mcts_improvements.py` → Script de la raíz para entrenar el agente, comparar su rendimiento y guardar los Q-values.

## Requisitos
- Python 3.10+
- Librerías necesarias: `numpy`

## Generar Q-values
Antes de competir en el torneo con su máximo potencial, es ideal generar el archivo:

`q_values.pkl`

Este archivo se crea ejecutando el script desde la **raíz del proyecto** 
```bash
python test_mcts_improvements.py
```
Al finalizar, generará un archivo llamado `q_values_trained.pkl`. Deberás moverlo a la carpeta de tu grupo (`groups/Group A/`) y renombrarlo a `q_values.pkl`.

Si el archivo no existe, el agente simplemente jugará usando puro MCTS sin predicciones pre-entrenadas.


