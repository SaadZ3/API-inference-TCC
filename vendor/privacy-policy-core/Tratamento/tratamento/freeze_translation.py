"""Congela somente a exportação conhecida do GOOGLETRANSLATE, sem avaliar fórmulas."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from tratamento.translation import check_structure, load_translation
from tratamento.workbook import verify_workbook, write_workbook


def extract_google_literal(formula: str, row: int) -> str:
    """Aceita apenas IFERROR(DUMMYFUNCTION(GOOGLETRANSLATE), literais concatenados)."""
    prefix = '=IFERROR(__xludf.DUMMYFUNCTION("GOOGLETRANSLATE(C' + str(row) + ',""en"",""pt"")"),'
    if not formula.startswith(prefix) or not formula.endswith(")"):
        raise ValueError(f"Fórmula inesperada na linha {row}.")
    tail = formula[len(prefix):-1]
    literal = r'"(?:[^"]|"")*"'
    if re.fullmatch(literal + r'(?:&' + literal + r')*', tail) is None:
        raise ValueError(f"Resultado não literal na linha {row}.")
    return "".join(part[1:-1].replace('""', '"') for part in re.findall(literal, tail))


def freeze(source: Path, destination: Path, reference_dir: Path) -> dict:
    """Preserva a origem e exige concordância entre literal e cache exportados."""
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("O destino deve ser um arquivo novo, diferente da origem.")
    reference_path = reference_dir / "segments.jsonl"
    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != manifest["segments_sha256"]:
        raise ValueError("Manifesto de origem divergente.")
    reference = pd.read_json(reference_path, lines=True, dtype={"segment_id": str, "policy_id": str, "annotation_policy_id": str, "split_group": str})
    frame, types = load_translation(source)
    check_structure(frame, reference, manifest["labels"])
    cached = load_workbook(source, read_only=True, data_only=True)
    try:
        sheet = cached["trechos"]
        headers = [cell.value for cell in next(sheet.iter_rows())]
        column = headers.index("texto_pt")
        values = list(sheet.iter_rows(min_row=2))
        converted = 0
        for index, record in enumerate(frame.to_dict("records")):
            if types[record["segment_id"]] != "f":
                continue
            literal = extract_google_literal(record["texto_pt"], index + 2)
            if not literal.strip() or literal != values[index][column].value:
                raise ValueError(f"Cache ausente ou diferente do resultado literal: {record['segment_id']}")
            frame.loc[index, "texto_pt"] = literal
            converted += 1
    finally:
        cached.close()
    write_workbook(frame, destination)
    verify_workbook(destination, frame)
    report = {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "destination": destination.name, "destination_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "converted_cells": converted, "method": "literal_equals_cached_value; no_formula_evaluation",
              "human_review_status": "pending"}
    destination.with_suffix(".provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixa valores da exportação do Google Planilhas")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path, default=Path("Tratamento/dados"))
    args = parser.parse_args()
    print(json.dumps(freeze(args.input, args.output, args.reference_dir), ensure_ascii=False, indent=2))
