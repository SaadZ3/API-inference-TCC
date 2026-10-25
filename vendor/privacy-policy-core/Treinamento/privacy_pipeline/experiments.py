"""Dados, procedência e relatórios comuns aos experimentos do treinamento."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tratamento.cli import ROOT, save_json
from tratamento.data import text_hash
from privacy_pipeline.metrics import classification_metrics, tune_thresholds
from tratamento.translation import check_structure, load_translation, validate_translation

LOGGER = logging.getLogger(__name__)


def load_experiment_data(config: dict, task: str) -> tuple[dict[str, pd.DataFrame], list[str], dict]:
    """Revalida a planilha em cada execução e preserva os splits do tratamento."""
    treatment = yaml.safe_load((ROOT / config["treatment_config"]).read_text(encoding="utf-8"))
    source_dir = ROOT / treatment["output_dir"]
    source = source_dir / "segments.jsonl"
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["segments_sha256"]:
        raise ValueError("Origem alterada em relação ao manifesto.")
    reference = pd.read_json(source, lines=True, dtype={"segment_id": str, "policy_id": str, "annotation_policy_id": str, "split_group": str})
    path = ROOT / config["translated_file"]
    frame, types = load_translation(path)
    labels = manifest["labels"]
    check_structure(frame, reference, labels)
    frame, validation = validate_translation(frame, types, treatment["translation"])
    if validation["rows_with_errors"]:
        raise ValueError(f"Tradução tem {validation['rows_with_errors']} erros impeditivos. Execute validate-translation.")
    review = None
    if config.get("human_review_record"):
        review = json.loads((ROOT / config["human_review_record"]).read_text(encoding="utf-8"))
        if review["translation_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("A revisão humana pertence a outra versão da tradução.")
        review_file = ROOT / config.get("human_review_file", review["review_file"])
        if review["review_file_sha256"] != hashlib.sha256(review_file.read_bytes()).hexdigest():
            raise ValueError("A planilha de revisão foi alterada; atualize o registro da revisão.")
        if review["status"] != config["human_review_status"]:
            raise ValueError("Status da revisão divergente da configuração.")
        validation["human_review_status"] = review["status"]
        validation["reviewed_segments"] = review["reviewed_segments"]
    if validation["rows_with_warnings"]:
        LOGGER.warning("%s traduções com avisos; revisão humana: %s.", validation["rows_with_warnings"], config["human_review_status"])
    before = len(frame)
    frame = frame[frame.eligible.astype(bool)].copy()
    if task == "multiclass":
        frame = frame[frame.n_labels == 1].copy()
    elif task != "multilabel":
        raise ValueError("task deve ser multilabel ou multiclass.")
    if frame.groupby("policy_id").split.nunique().max() != 1:
        raise ValueError("Vazamento de política entre splits.")
    if frame.groupby("text_sha256").split.nunique().max() != 1:
        raise ValueError("Texto original duplicado entre splits.")
    frame["translation_sha256"] = frame.texto_pt.map(text_hash)
    duplicates = int((frame.groupby("translation_sha256").split.nunique() > 1).sum())
    if duplicates:
        LOGGER.warning("%s textos traduzidos idênticos entre splits; registrado para análise.", duplicates)
    splits = {name: frame.loc[frame.split == name].sort_values("segment_id").reset_index(drop=True)
              for name in ["train", "validation", "test"]}
    if any(len(rows) == 0 for rows in splits.values()):
        raise ValueError("Split vazio após seleção da tarefa.")
    metadata = {"task": task, "labels": labels, "seed": config["seed"], "config": config,
                "translation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_segments_sha256": manifest["segments_sha256"],
                "human_review_status": config["human_review_status"], "translation_validation": validation,
                "human_review_record": review,
                "excluded_segments": before - len(frame), "translated_duplicates_across_splits": duplicates,
                "multiclass_policy": "somente segmentos com exatamente um rótulo; população diferente" if task == "multiclass" else None,
                "splits": {name: {"segments": len(rows), "policies": int(rows.policy_id.nunique()),
                                   "label_support": {label: int(rows[label].sum()) for label in labels}}
                           for name, rows in splits.items()},
                "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "transformers", "torch", "datasets", "accelerate"]}}
    return splits, labels, metadata


def targets(frame: pd.DataFrame, labels: list[str], task: str) -> np.ndarray:
    """Retorna indicadores ou índice único sem escolher rótulos arbitrariamente."""
    matrix = frame[labels].to_numpy(dtype=int)
    if task == "multiclass":
        if not (matrix.sum(axis=1) == 1).all():
            raise ValueError("Multiclass recebeu trecho com zero ou múltiplos rótulos.")
        return matrix.argmax(axis=1)
    return matrix


def create_run(config: dict, run_name: str) -> Path:
    """Impede sobrescrita de experimentos e escolha de caminhos externos no nome."""
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run-name deve ser um nome simples de diretório.")
    path = ROOT / config["output_dir"] / run_name
    path.mkdir(parents=True, exist_ok=False)
    return path


def evaluate_scores(splits: dict[str, pd.DataFrame], labels: list[str], task: str,
                    scores: dict[str, np.ndarray], probabilities: bool, output: Path,
                    language: str) -> dict:
    """Ajusta limiares na validação e só então avalia e exporta o teste."""
    default = 0.5 if probabilities else 0.0
    if task == "multilabel":
        thresholds, statuses = tune_thresholds(targets(splits["validation"], labels, task), scores["validation"], default, probabilities)
        save_json({"labels": labels, "thresholds": thresholds.tolist(), "statuses": statuses,
                   "score_kind": "probability" if probabilities else "decision_margin", "fitted_on": "validation"}, output / "thresholds.json")
    else:
        thresholds = None
    reports = {}
    for name in ["validation", "test"]:
        truth = targets(splits[name], labels, task)
        pred = (scores[name] >= thresholds).astype(int) if thresholds is not None else scores[name].argmax(axis=1)
        report = classification_metrics(truth, pred, labels, task)
        if task == "multilabel":
            report["default_threshold_metrics"] = classification_metrics(truth, (scores[name] >= default).astype(int), labels, task)
        report["threshold_selection_bias"] = name == "validation" and task == "multilabel"
        reports[name] = report
        export_errors(splits[name], truth, pred, scores[name], labels, task, language,
                      probabilities, output / f"{name}_predictions.jsonl", output / f"{name}_worst_errors.jsonl")
    save_json(reports, output / "metrics.json")
    table = ["# Resultados executados", "", "| Conjunto | F1-macro | F1-micro | Kappa |", "|---|---:|---:|---:|"]
    for name, report in reports.items():
        kappa = "indefinido" if report["cohen_kappa"] is None else f"{report['cohen_kappa']:.6f}"
        table.append(f"| {name} | {report['f1_macro']:.6f} | {report['f1_micro']:.6f} | {kappa} |")
    table.extend(["", "A validação foi usada para selecionar thresholds e não é estimativa independente.",
                  "Consulte run.json para procedência, revisão da tradução, pesos e exclusões.",
                  "Matrizes e métricas por classe estão em metrics.json. Kappas indefinidos não viram zero.", ""])
    (output / "RESULTADOS.md").write_text("\n".join(table), encoding="utf-8")
    return reports


def export_errors(frame: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, scores: np.ndarray,
                  labels: list[str], task: str, language: str, probabilities: bool,
                  predictions_path: Path, errors_path: Path) -> None:
    """Ordena erros por número de rótulos errados e confiança/margem da decisão."""
    records = []
    for index, row in enumerate(frame.to_dict("records")):
        if task == "multilabel":
            actual_ids, predicted_ids = np.flatnonzero(truth[index]), np.flatnonzero(pred[index])
            mismatches = int(np.count_nonzero(truth[index] != pred[index]))
        else:
            actual_ids, predicted_ids = [truth[index]], [pred[index]]
            mismatches = int(truth[index] != pred[index])
        severity = float(np.mean(np.abs(scores[index] - truth[index]))) if task == "multilabel" and probabilities else float(np.max(scores[index]))
        records.append({"segment_id": row["segment_id"], "policy_id": row["policy_id"], "split": row["split"],
                        "text": row[f"texto_{language}"], "texto_en": row["texto_en"], "texto_pt": row["texto_pt"],
                        "true_labels": [labels[i] for i in actual_ids], "predicted_labels": [labels[i] for i in predicted_ids],
                        "scores": {label: float(scores[index, i]) for i, label in enumerate(labels)},
                        "score_kind": "probability" if probabilities else "decision_margin", "wrong_labels": mismatches,
                        "ranking_score": severity})
    def write(path: Path, values: list[dict]) -> None:
        path.write_text("".join(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n" for value in values), encoding="utf-8")
    write(predictions_path, records)
    errors = sorted([record for record in records if record["wrong_labels"]],
                    key=lambda record: (-record["wrong_labels"], -record["ranking_score"], record["segment_id"]))
    write(errors_path, errors[:100])
