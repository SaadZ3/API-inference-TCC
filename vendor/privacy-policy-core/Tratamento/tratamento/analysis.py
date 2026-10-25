"""Estatísticas descritivas e contagem real de subwords do BERTimbau."""

from __future__ import annotations

import json
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def summarize_lengths(lengths: list[int], candidates: list[int]) -> dict[str, Any]:
    """Calcula quantis e cobertura, incluindo tokens especiais na contagem."""
    if not lengths:
        raise ValueError("Não há comprimentos para analisar.")
    series = pd.Series(lengths)
    return {
        "count": len(lengths), "min": int(series.min()), "max": int(series.max()),
        "mean": float(series.mean()),
        "quantiles": {str(q): float(series.quantile(q)) for q in [0.5, 0.9, 0.95, 0.99]},
        "coverage": {str(n): float((series <= n).mean()) for n in candidates},
        "truncated_count": {str(n): int((series > n).sum()) for n in candidates},
    }


def tokenize_lengths(frame: pd.DataFrame, column: str, config: dict, root: Path) -> tuple[list[int], dict]:
    """Baixa somente tokenizer/configuração; não carrega pesos do modelo."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["name"], revision=config["revision"],
        cache_dir=str(root / config["cache_dir"]), use_fast=True,
    )
    lengths: list[int] = []
    for start in range(0, len(frame), 128):
        batch = frame[column].iloc[start:start + 128].tolist()
        encoded = tokenizer(batch, truncation=False, padding=False, add_special_tokens=True,
                            return_attention_mask=False, return_token_type_ids=False,
                            verbose=False)
        lengths.extend(map(len, encoded["input_ids"]))
    revision = tokenizer.init_kwargs.get("_commit_hash")
    if revision is None and re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        revision = config["revision"]
    metadata = {"name": config["name"], "requested_revision": config["revision"],
                "resolved_revision": revision,
                "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest(),
                "text_column": column, "special_tokens_included": True,
                "truncation": False}
    return lengths, metadata


def exploratory_report(frame: pd.DataFrame, labels: list[str], source: dict,
                       token_metadata: dict | None, config: dict) -> dict:
    """Agrega suporte por classe e split sem calcular métricas de modelo."""
    eligible = frame[frame.eligible]
    support = {label: int(eligible[label].sum()) for label in labels}
    report = {
        **source, "segments": len(frame), "eligible_segments": len(eligible),
        "unlabeled_segments": int((frame.n_labels == 0).sum()),
        "multilabel_segments": int((eligible.n_labels > 1).sum()),
        "multilabel_fraction": float((eligible.n_labels > 1).mean()),
        "union_multilabel_segments": int((frame.n_labels_union > 1).sum()),
        "majority_multilabel_segments": int((frame.n_labels_majority > 1).sum()),
        "majority_unlabeled_segments": int((frame.n_labels_majority == 0).sum()),
        "label_support": support,
        "imbalance_max_min": max(support.values()) / min(support.values()),
        "n_labels_distribution": {str(k): int(v) for k, v in frame.n_labels.value_counts().sort_index().items()},
        "exact_duplicate_text_groups": int((frame.groupby("text_sha256").size() > 1).sum()),
        "duplicate_text_groups_across_policies": int((frame.groupby("text_sha256").policy_id.nunique() > 1).sum()),
        "cross_split_duplicate_texts": int((frame.groupby("text_sha256").split.nunique() > 1).sum()),
        "split_groups": int(frame.split_group.nunique()),
        "split": {}, "tokenizer": token_metadata,
    }
    for name, rows in frame.groupby("split", sort=True):
        report["split"][name] = {
            "policies": int(rows.policy_id.nunique()), "segments": len(rows),
            "eligible_segments": int(rows.eligible.sum()),
            "label_support": {label: int(rows.loc[rows.eligible, label].sum()) for label in labels},
        }
    if "n_tokens_en" in frame:
        report["tokens_en"] = summarize_lengths(frame.n_tokens_en.tolist(), config["lengths"])
        train_lengths = eligible.loc[eligible.split == "train", "n_tokens_en"].tolist()
        train = summarize_lengths(train_lengths, config["lengths"])
        report["tokens_en_train"] = train
        feasible = [n for n in config["lengths"] if n <= 512 and train["coverage"][str(n)] >= config["target_coverage"]]
        report["provisional_max_length"] = min(feasible) if feasible else 512
        report["target_coverage_reached"] = bool(feasible)
    else:
        report["token_analysis_status"] = "Não executada; não substituída por contagem de palavras."
    return report


def write_report(report: dict, path: Path) -> None:
    """Escreve relatório legível com os valores efetivamente calculados."""
    lines = ["# tratamento — análise exploratória", "",
             f"Políticas: {report['policies']}. Trechos: {report['segments']}. Anotações: {report['annotations']}.",
             f"Regra: pelo menos {report['min_votes']} voto(s) por categoria. Elegíveis: {report['eligible_segments']}.",
             f"Trechos multi-label: {report['multilabel_segments']} ({report['multilabel_fraction']:.2%}).",
             f"União: {report['union_multilabel_segments']} multi-label; consenso ≥2: {report['majority_multilabel_segments']}.",
             f"Sem categoria com consenso ≥2: {report['majority_unlabeled_segments']}.", "",
             "Recomendação: tarefa multi-label. A união pode incorporar desacordos; votos permanecem auditáveis.",
             "", "## Classes", "", "| Categoria | Total | Treino | Validação | Teste |", "|---|---:|---:|---:|---:|"]
    for label, count in report["label_support"].items():
        values = [report["split"][split]["label_support"][label] for split in ["train", "validation", "test"]]
        lines.append(f"| {label} | {count} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(["", f"Razão de suporte maior/menor: {report['imbalance_max_min']:.2f}.",
                  "As contagens são presenças por trecho; sua soma excede o número de trechos.",
                  "", "## Splits", "", "| Conjunto | Políticas | Trechos | Elegíveis |", "|---|---:|---:|---:|"])
    for name, values in report["split"].items():
        lines.append(f"| {name} | {values['policies']} | {values['segments']} | {values['eligible_segments']} |")
    lines.extend(["", f"Componentes de políticas: {report['split_groups']}.",
                  f"Textos repetidos entre políticas: {report['duplicate_text_groups_across_policies']}; atravessando splits: {report['cross_split_duplicate_texts']}.",
                  "Políticas conectadas por trechos iguais ficam no mesmo conjunto. As proporções são aproximadas.",
                  "Similaridade semântica e quase duplicatas não são detectadas por este controle.",
                  "", "## Tokens", ""])
    if "tokens_en" in report:
        lines.extend(["Tokenizer BERTimbau aplicado ao inglês, sem truncamento e com tokens especiais.",
                      "**Análise provisória:** repetir em português após a tradução; inglês não determina o comprimento final.",
                      "", "| Limite | Cobertura total | Trechos acima | Cobertura no treino |", "|---|---:|---:|---:|"])
        for limit, coverage in report["tokens_en"]["coverage"].items():
            lines.append(f"| {limit} | {coverage:.2%} | {report['tokens_en']['truncated_count'][limit]} | {report['tokens_en_train']['coverage'][limit]:.2%} |")
        lines.extend(["", f"Quantis (total): {json.dumps(report['tokens_en']['quantiles'])}.",
                      f"Sugestão provisória baseada apenas no treino: max_length={report['provisional_max_length']}.",
                      f"Meta de cobertura atingida: {report['target_coverage_reached']}.",
                      "Acima de 512, avaliar janelas ou fragmentação na treinamento; não descartar silenciosamente o final dos trechos."])
    else:
        lines.append(report["token_analysis_status"])
    lines.extend(["", "## Limitações e procedência", "",
                  f"IDs internos divergentes do prefixo do arquivo: {report['policy_id_mismatches']} políticas.",
                  "As fontes são annotations/, sanitized_policies/ e documentation/categories-july30.xml.",
                  "Os spans com índices problemáticos não são usados; a unidade é o segmento completo.",
                  "Não há split oficial nesta distribuição local. O manifesto fixa um split próprio com seed 42 (ou a seed configurada).",
                  "Nenhum modelo foi treinado ou avaliado nesta fase. Nenhuma métrica preditiva foi estimada.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
