"""Métricas e limiares por classe, com tratamento explícito de kappa indefinido."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix, precision_recall_fscore_support


def kappa_from_confusion(matrix: np.ndarray) -> float | None:
    """Calcula Cohen's kappa; retorna None quando a concordância casual é um."""
    matrix = np.asarray(matrix, dtype=float)
    total = matrix.sum()
    if total <= 0:
        return None
    observed = np.trace(matrix) / total
    chance = float(matrix.sum(axis=0) @ matrix.sum(axis=1)) / total ** 2
    return None if np.isclose(chance, 1.0) else float((observed - chance) / (1.0 - chance))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], task: str) -> dict:
    """Usa a ordem fixa de categorias; classes ausentes recebem F1 zero."""
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if truth.shape != pred.shape or len(truth) == 0:
        raise ValueError("Verdade e previsão devem ter formatos iguais e ser não vazias.")
    indices = np.arange(len(labels))
    if task == "multilabel":
        if truth.ndim != 2 or truth.shape[1] != len(labels) or not np.isin(truth, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
            raise ValueError("Matrizes multi-label devem ser binárias, com uma coluna por classe.")
        matrices = multilabel_confusion_matrix(truth, pred, labels=indices)
        kappas = [kappa_from_confusion(matrix) for matrix in matrices]
        defined = [value for value in kappas if value is not None]
        kappa = float(np.mean(defined)) if defined else None
        confusion = {label: matrices[i].tolist() for i, label in enumerate(labels)}
    elif task == "multiclass":
        if truth.ndim != 1 or not np.isin(truth, indices).all() or not np.isin(pred, indices).all():
            raise ValueError("Índices multiclass inválidos.")
        matrix = confusion_matrix(truth, pred, labels=indices)
        kappa = kappa_from_confusion(matrix)
        kappas = [None] * len(labels)
        confusion = matrix.tolist()
    else:
        raise ValueError("Tarefa inválida.")
    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=indices, zero_division=0)
    return {
        "n_samples": len(truth), "f1_macro": float(f1_score(truth, pred, labels=indices, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(truth, pred, labels=indices, average="micro", zero_division=0)),
        "cohen_kappa": kappa,
        "kappa_definition": "média dos kappas binários definidos" if task == "multilabel" else "kappa multiclass",
        "undefined_kappa_classes": [label for label, value in zip(labels, kappas) if value is None] if task == "multilabel" else [],
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                              "support": int(support[i]), "kappa": kappas[i]} for i, label in enumerate(labels)},
        "confusion_matrix": confusion,
        "confusion_order": "[[TN, FP], [FN, TP]] por classe" if task == "multilabel" else "linhas=reais, colunas=previstas, ordem=labels",
    }


def tune_thresholds(y_validation: np.ndarray, scores_validation: np.ndarray,
                    default: float, probabilities: bool) -> tuple[np.ndarray, list[str]]:
    """Maximiza F1 por classe exclusivamente na validação, com desempate estável."""
    truth, scores = np.asarray(y_validation), np.asarray(scores_validation)
    if truth.shape != scores.shape or truth.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Formato ou scores inválidos para ajuste de thresholds.")
    thresholds = np.full(truth.shape[1], default, dtype=float)
    statuses = []
    for index in range(truth.shape[1]):
        if len(np.unique(truth[:, index])) < 2:
            statuses.append("default: validação sem positivos ou negativos")
            continue
        grid = np.linspace(0.05, 0.95, 19) if probabilities else np.quantile(scores[:, index], np.linspace(0, 1, 101))
        grid = np.unique(np.append(grid, default))
        best = max(grid, key=lambda threshold: (
            f1_score(truth[:, index], scores[:, index] >= threshold, zero_division=0),
            -abs(threshold - default), -threshold))
        thresholds[index] = float(best)
        statuses.append("ajustado na validação")
    return thresholds, statuses


def imbalance_weights(y_train: np.ndarray, task: str, n_classes: int) -> np.ndarray:
    """Calcula pesos somente no treino e rejeita classes sem exemplos."""
    if task == "multilabel":
        positive = y_train.sum(axis=0)
        negative = len(y_train) - positive
        if (positive == 0).any() or (negative == 0).any():
            raise ValueError("Classe de treino sem positivos ou negativos.")
        return (negative / positive).astype(np.float32)
    counts = np.bincount(y_train, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError("Classe de treino sem exemplos no subconjunto multiclass.")
    return (len(y_train) / (n_classes * counts)).astype(np.float32)
