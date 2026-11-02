"""Comandos reproduzíveis do tratamento."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from tratamento.analysis import exploratory_report, summarize_lengths, tokenize_lengths, write_report
from tratamento.data import load_corpus
from tratamento.splitting import assign_splits
from tratamento.translation import check_structure, load_translation, review_sample, validate_translation
from tratamento.workbook import verify_workbook, write_workbook

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def save_json(value: object, path: Path) -> None:
    """Serializa resultados e configuração em UTF-8."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare(config: dict, skip_tokens: bool, force: bool) -> None:
    """Gera corpus, planilha, manifesto fixo e análise exploratória."""
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    workbook_path = output / "opp115_translation.xlsx"
    if any((output / name).exists() for name in ["opp115_translation.xlsx", "segments.jsonl", "manifest.json"]) and not force:
        raise ValueError("Artefatos já existem. Use --force só para regenerar a origem; guarde a tradução em outro arquivo.")
    source_root = ROOT / config["dataset_root"]
    frame, labels, source = load_corpus(source_root, int(config["min_votes"]))
    frame = assign_splits(frame, labels, config["split"], int(config["seed"]))
    metadata = None
    if not skip_tokens:
        LOGGER.info("Contando tokens reais de %s trechos.", len(frame))
        lengths, metadata = tokenize_lengths(frame, "texto_en", config["tokenizer"], ROOT)
        frame["n_tokens_en"] = lengths
    report = exploratory_report(frame, labels, source, metadata, config["tokenizer"])
    write_workbook(frame, workbook_path)
    verify_workbook(workbook_path, frame)
    frame.to_json(output / "segments.jsonl", orient="records", lines=True, force_ascii=False)
    frame[["segment_id", "policy_id", "split", "split_group", "eligible", "text_sha256"]].to_csv(output / "split_manifest.csv", index=False)
    source_paths = sorted(set(frame.source_file) | set(frame.annotation_file) | {"documentation/categories-july30.xml"})
    manifest = {"config": config, "labels": labels, "tokenizer": metadata,
                "sources_sha256": {path: hashlib.sha256((source_root / path).read_bytes()).hexdigest() for path in source_paths},
                "segments_sha256": hashlib.sha256((output / "segments.jsonl").read_bytes()).hexdigest(),
                "versions": {name: importlib.metadata.version(name) for name in ["pandas", "openpyxl", "PyYAML", "transformers"]}}
    save_json(manifest, output / "manifest.json")
    save_json(report, output / "eda.json")
    save_json({str(index): label for index, label in enumerate(labels)}, output / "label_mapping.json")
    write_report(report, output / "EDA.md")
    LOGGER.info("tratamento preparada: %s trechos, %s multi-label. Planilha: %s", len(frame), report["multilabel_segments"], workbook_path)


def read_reference(output: Path) -> tuple[dict, pd.DataFrame]:
    """Recebe a pasta dos dados e retorna manifesto e segmentos após conferir a integridade."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    source_path = output / "segments.jsonl"
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest["segments_sha256"]:
        raise ValueError("segments.jsonl diverge do manifesto original.")
    reference = pd.read_json(source_path, lines=True, dtype={"segment_id": str, "policy_id": str, "annotation_policy_id": str, "split_group": str})
    return manifest, reference


def validate(config: dict, path: Path, skip_tokens: bool) -> int:
    """Confere tradução contra a origem e exporta amostra de revisão."""
    output = ROOT / config["output_dir"]
    manifest, reference = read_reference(output)
    frame, types = load_translation(path)
    labels = manifest["labels"]
    check_structure(frame, reference, labels)
    checked, summary = validate_translation(frame, types, config["translation"])
    validation_dir = output / "translation_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    checked.to_json(validation_dir / "validation_details.jsonl", orient="records", lines=True, force_ascii=False)
    # TSV evita ambiguidades de delimitador para a revisão manual.
    review_sample(checked, labels, config["translation"], int(config["seed"])).to_csv(validation_dir / "manual_review.tsv", sep="\t", index=False)
    summary["input_file"] = path.name
    summary["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not summary["rows_with_errors"] and not skip_tokens:
        lengths, metadata = tokenize_lengths(checked, "texto_pt", config["tokenizer"], ROOT)
        checked["n_tokens_pt"] = lengths
        summary["tokenizer"] = metadata
        summary["tokens_pt"] = summarize_lengths(lengths, config["tokenizer"]["lengths"])
        train = checked.loc[(checked.split == "train") & checked.eligible.astype(bool), "n_tokens_pt"].tolist()
        summary["tokens_pt_train"] = summarize_lengths(train, config["tokenizer"]["lengths"])
        candidates = [n for n in config["tokenizer"]["lengths"] if n <= 512 and summary["tokens_pt_train"]["coverage"][str(n)] >= config["tokenizer"]["target_coverage"]]
        summary["suggested_max_length"] = min(candidates) if candidates else 512
        summary["target_coverage_reached"] = bool(candidates)
        checked[["segment_id", "n_tokens_pt"]].to_csv(validation_dir / "token_lengths_pt.csv", index=False)
    else:
        summary["token_analysis_status"] = "Não executada: tradução incompleta ou --skip-tokens."
    save_json(summary, validation_dir / "summary.json")
    LOGGER.info("Validação: %s erros, %s avisos; revisão humana pendente.", summary["rows_with_errors"], summary["rows_with_warnings"])
    return 2 if summary["rows_with_errors"] else 0


def main() -> int:
    """Lê os argumentos do terminal e retorna o código de saída do tratamento, sem treinar."""
    parser = argparse.ArgumentParser(description="tratamento do OPP-115 / BERTimbau")
    parser.add_argument("command", choices=["prepare", "validate-translation"])
    parser.add_argument("--config", type=Path, default=ROOT / "Tratamento/config.yaml")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--skip-tokens", action="store_true", help="Executa sem rede; registra análise de tokens como pendente.")
    parser.add_argument("--force", action="store_true", help="Regenera artefatos de origem existentes.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        if args.command == "prepare":
            prepare(config, args.skip_tokens, args.force)
            return 0
        if args.input is None:
            parser.error("validate-translation requer --input.")
        return validate(config, args.input, args.skip_tokens)
    except (ValueError, FileNotFoundError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
