"""Validação das previsões, escala e identificação do conteúdo."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any


LABELS = (
    "First Party Collection/Use", "Third Party Sharing/Collection", "User Choice/Control",
    "User Access, Edit and Deletion", "Data Retention", "Data Security", "Policy Change",
    "Do Not Track", "International and Specific Audiences", "Other",
)

SCORE_SCALE = "continuous_0_3"
# Mesma tolerância do backend para resíduos de cálculo nas fronteiras das faixas.
BOUNDARY_TOLERANCE = 1e-12


def classify_percentage(value: float) -> int:
    """Converte o percentual sem deixar resíduos numéricos alterarem a faixa."""
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("O percentual deve ser finito e estar entre 0 e 100.")
    return sum(value > limit + BOUNDARY_TOLERANCE for limit in (25, 50, 75))


def continuous_score(value: float) -> float:
    """Preserva o crédito parcial, sem aplicar as faixas do histórico."""
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("O percentual deve ser finito e estar entre 0 e 100.")
    return value * 3 / 100


def content_hash(value: Any) -> str:
    """Identifica o conteúdo JSON, independentemente da indentação do arquivo."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_policy(policy: dict, predictions: bool = True) -> None:
    """Recusa entradas ambíguas, em vez de inferir completude ou rótulos."""
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("A entrada deve usar schema_version 1.")
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"].strip():
        raise ValueError("Informe policy_id.")
    if policy.get("language") != "pt" or type(policy.get("complete")) is not bool:
        raise ValueError("Informe language pt e complete booleano.")
    applicability = policy.get("applicability", {})
    if not isinstance(applicability, dict) or any(type(v) is not bool for v in applicability.values()):
        raise ValueError("applicability deve mapear critérios opcionais a booleanos.")
    segments = policy.get("segments")
    if not isinstance(segments, list):
        raise ValueError("segments deve ser uma lista.")
    ids = []
    for segment in segments:
        if not isinstance(segment, dict) or not all(
            isinstance(segment.get(key), str) and segment[key].strip() for key in ("segment_id", "text")
        ):
            raise ValueError("Cada trecho exige segment_id e text não vazios.")
        ids.append(segment["segment_id"])
        if predictions:
            labels, probs = segment.get("labels"), segment.get("probabilities")
            if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
                raise ValueError("labels deve ser uma lista de categorias.")
            if len(labels) != len(set(labels)) or not set(labels) <= set(LABELS):
                raise ValueError("Rótulos duplicados ou desconhecidos.")
            if not isinstance(probs, dict) or set(probs) != set(LABELS):
                raise ValueError("Informe probabilidades para as dez categorias.")
            if any(not _number(v) or not 0 <= v <= 1 for v in probs.values()):
                raise ValueError("Probabilidades devem ser finitas, entre zero e um.")
            if type(segment.get("truncated")) is not bool:
                raise ValueError("Cada previsão deve informar truncated booleano.")
            if type(segment.get("n_tokens")) is not int or segment["n_tokens"] <= 0:
                raise ValueError("Cada previsão deve informar n_tokens positivo.")
    if len(ids) != len(set(ids)):
        raise ValueError("segment_id deve ser único na política.")
    if predictions and policy.get("model") is not None:
        _validate_model_predictions(policy)


def _validate_model_predictions(policy: dict) -> None:
    """Confere metadados e decisões quando existe proveniência do adaptador."""
    model = policy["model"]
    if not isinstance(model, dict):
        raise ValueError("Metadados de modelo inválidos.")
    settings, thresholds = model.get("inference", {}), model.get("thresholds", {})
    if not isinstance(settings, dict) or not isinstance(thresholds, dict):
        raise ValueError("Metadados de inferência inválidos.")
    if (settings.get("task") != "multilabel" or settings.get("language") != "pt"
            or type(settings.get("max_length")) is not int or settings["max_length"] <= 2):
        raise ValueError("Modelo incompatível com a avaliação.")
    values = thresholds.get("thresholds")
    if (thresholds.get("labels") != list(LABELS) or thresholds.get("score_kind") != "probability"
            or thresholds.get("fitted_on") != "validation" or not isinstance(values, list)
            or len(values) != len(LABELS)
            or any(not _number(v) or not 0 <= v <= 1 for v in values)):
        raise ValueError("Thresholds de proveniência inválidos.")
    # O adaptador compara tensores float32; preservamos essa precisão sem PyTorch.
    values = [struct.unpack("f", struct.pack("f", v))[0] for v in values]
    for segment in policy["segments"]:
        expected = {label for label, value in zip(LABELS, values, strict=True)
                    if segment["probabilities"][label] >= value}
        if expected != set(segment["labels"]):
            raise ValueError("Rótulos divergem das probabilidades e thresholds registrados.")
        if segment["truncated"] != (segment["n_tokens"] > settings["max_length"]):
            raise ValueError("Indicador de truncamento diverge dos metadados.")


