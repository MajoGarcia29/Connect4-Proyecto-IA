"""
PImp — Policy Improvement step de TBOPI.

sample_action_from_inner_stats:
    Dado q_local y n_local acumulados por los trials internos,
    construye una distribución de probabilidad sobre las acciones
    legales y samplea la acción a jugar.

    score(a) = (max(q̂(s,a), 0) + ε) · (N(a) + 1)
    p(a)     = score(a) / Σ score(b)
"""

import numpy as np

def sample_action_from_inner_stats(
    q_local: dict[int, float],
    n_local: dict[int, int],
    legal_actions: list[int],
    *,
    use_q: bool = True,
    use_counts: bool = True,
    rng: np.random.RandomState,
    eps: float = 1e-6,
) -> tuple[int, dict[int, float]]:
    """
    Paso PImp de TBOPI.

    Garantías:
    - p(a) > 0 para toda acción legal (por ε).
    - Valores negativos no penalizan (clip a 0).
    - Acciones más visitadas tienen más peso (confianza).
    """
    actions = list(legal_actions)
    if not actions:
        raise ValueError("legal_actions está vacío.")

    scores = []
    for a in actions:
        q_val  = q_local.get(a, 0.0) if use_q     else 0.0
        n_val  = n_local.get(a, 0)   if use_counts else 0
        scores.append((max(q_val, 0.0) + eps) * (n_val + 1))

    total      = sum(scores)
    probs_list = [s / total for s in scores]
    probs      = {a: p for a, p in zip(actions, probs_list)}
    chosen     = rng.choice(actions, p=probs_list)

    return int(chosen), probs
