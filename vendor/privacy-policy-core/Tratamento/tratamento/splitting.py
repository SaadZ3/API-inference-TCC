"""Separação determinística por política e componentes de textos repetidos."""

from __future__ import annotations

import random
import numpy as np
import pandas as pd


def policy_groups(frame: pd.DataFrame, group_duplicates: bool) -> dict[str, str]:
    """Agrupa transitivamente políticas que compartilham trechos normalizados."""
    parent = {str(policy): str(policy) for policy in sorted(frame.policy_id.unique())}

    def find(policy: str) -> str:
        while parent[policy] != policy:
            parent[policy] = parent[parent[policy]]
            policy = parent[policy]
        return policy

    if group_duplicates:
        for _, rows in frame.groupby("text_sha256", sort=True):
            policies = sorted(rows.policy_id.unique())
            for policy in policies[1:]:
                left, right = find(str(policies[0])), find(str(policy))
                parent[max(left, right)] = min(left, right)
    return {policy: find(policy) for policy in parent}


def assign_splits(frame: pd.DataFrame, labels: list[str], config: dict, seed: int) -> pd.DataFrame:
    """Escolhe entre splits aleatórios usando apenas tamanhos e suportes de classe."""
    names = ["train", "validation", "test"]
    ratios = np.array([config[name] for name in names], dtype=float)
    if (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("Proporções positivas de split devem somar 1.")
    candidates = int(config["candidates"])
    if candidates < 1:
        raise ValueError("split.candidates deve ser positivo.")
    result = frame.copy()
    mapping = policy_groups(result, bool(config["group_exact_duplicates"]))
    result["split_group"] = result.policy_id.map(mapping)
    groups = sorted(set(mapping.values()))
    if len(groups) < 3:
        raise ValueError("Menos de três grupos independentes para os splits.")
    active = result[result.n_labels > 0]
    grouped = active.groupby("split_group")[labels].sum().reindex(groups, fill_value=0)
    counts = active.groupby("split_group").size().reindex(groups, fill_value=0).to_numpy()
    policy_counts = result.groupby("split_group").policy_id.nunique().reindex(groups).to_numpy()
    matrix = np.column_stack([policy_counts, counts, grouped.to_numpy()])
    totals = matrix.sum(axis=0)
    if (totals[2:] == 0).any():
        raise ValueError("Há categoria sem exemplos após a consolidação.")
    rng = random.Random(seed)
    best_score = float("inf")
    best: np.ndarray | None = None
    # Sem consulta a previsões ou métricas: estratificação aproximada por grupos.
    for _ in range(candidates):
        assignment = np.array(rng.choices(range(3), weights=ratios, k=len(groups)))
        support = np.array([matrix[assignment == i].sum(axis=0) for i in range(3)])
        missing = int((support[:, 2:] == 0).sum())
        deviations = support / np.maximum(totals, 1) - ratios[:, None]
        score = missing * 1000 + float(np.mean(deviations ** 2))
        if score < best_score:
            best, best_score = assignment, score
    assert best is not None
    if any((matrix[best == i].sum(axis=0)[2:] == 0).any() for i in range(3)):
        raise ValueError("Não foi possível cobrir todas as classes nos três splits; revise os grupos.")
    split_map = {group: names[int(best[i])] for i, group in enumerate(groups)}
    result["split"] = result.split_group.map(split_map)
    result["eligible"] = result.n_labels > 0
    return result
