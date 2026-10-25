"""Validação estrutural e heurística da tradução, com revisão humana explícita."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from tratamento.data import text_hash

ERROR_PREFIXES = ("#REF!", "#VALUE!", "#VALOR!", "#N/A", "#N/D", "#NAME?", "#NOME?",
                  "#ERROR!", "#ERRO!", "#DIV/0!", "#NUM!", "#NULL!", "#SPILL!", "#CALC!",
                  "LOADING...", "CARREGANDO...")
URL_PATTERN = re.compile(r"https?://[^\s<>]+|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")


def load_translation(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Lê valores e tipos das células para não aceitar fórmula com cache antigo."""
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        if "trechos" not in book.sheetnames:
            raise ValueError("A aba trechos não existe.")
        rows = book["trechos"].iter_rows()
        columns = [str(cell.value or "") for cell in next(rows)]
        if len(columns) != len(set(columns)) or "" in columns:
            raise ValueError("Cabeçalhos duplicados ou vazios.")
        required = {"segment_id", "policy_id", "texto_en", "texto_pt", "labels", "n_labels", "split"}
        if not required <= set(columns):
            raise ValueError(f"Colunas ausentes: {sorted(required - set(columns))}")
        records = []
        cell_types = {}
        for row in rows:
            record = {name: (cell.value if cell.value is not None else "") for name, cell in zip(columns, row, strict=True)}
            if not any(value != "" for value in record.values()):
                continue
            record["segment_id"] = str(record["segment_id"])
            record["policy_id"] = str(record["policy_id"])
            records.append(record)
            cell_types[record["segment_id"]] = row[columns.index("texto_pt")].data_type
        return pd.DataFrame(records, columns=columns), cell_types
    finally:
        book.close()


def check_structure(frame: pd.DataFrame, reference: pd.DataFrame, labels: list[str]) -> None:
    """Confere IDs e metadados imutáveis pelo manifesto, mesmo após reordenação."""
    if frame.segment_id.duplicated().any():
        raise ValueError("Há segment_id duplicado na tradução.")
    if set(frame.segment_id) != set(reference.segment_id):
        raise ValueError("Há trechos ausentes ou adicionais em relação ao manifesto.")
    expected = reference.set_index("segment_id").loc[frame.segment_id]
    observed = frame.set_index("segment_id")
    immutable = [column for column in reference.columns if column not in {"segment_id", "texto_pt"}]
    missing = set(immutable) - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas imutáveis ausentes: {sorted(missing)}")
    for column in immutable:
        if pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(expected[column]):
            # O Sheets exporta inteiros como floats; comparar valores exatos, sem arredondar.
            actual = pd.to_numeric(observed[column], errors="coerce")
            equal = actual.notna().all() and actual.eq(expected[column]).all()
        else:
            equal = observed[column].astype(str).equals(expected[column].astype(str))
        if not equal:
            raise ValueError(f"Coluna original alterada: {column}")
    for row in frame.to_dict("records"):
        selected = json.loads(row["labels"])
        if not isinstance(selected, list) or len(selected) != len(set(selected)) or not set(selected) <= set(labels):
            raise ValueError(f"Labels inválidos: {row['segment_id']}")
        if int(row["n_labels"]) != len(selected) or any(int(row[label]) != int(label in selected) for label in labels):
            raise ValueError(f"Multi-hot inconsistente: {row['segment_id']}")
        if text_hash(row["texto_en"]) != row["text_sha256"]:
            raise ValueError(f"Hash do texto original divergente: {row['segment_id']}")


def translation_issues(source: str, target: Any, cell_type: str, config: dict) -> tuple[list[str], list[str], float | None]:
    """Sinaliza suspeitas; tamanho e termos não comprovam equivalência semântica."""
    errors: list[str] = []
    warnings: list[str] = []
    if target is None or not str(target).strip():
        return ["empty_translation"], warnings, None
    if cell_type == "f" or str(target).lstrip().startswith("="):
        return ["formula_not_pasted_as_values"], warnings, None
    if cell_type == "e" or str(target).strip().upper().startswith(ERROR_PREFIXES):
        return ["spreadsheet_error"], warnings, None
    if not isinstance(target, str):
        return ["non_text_translation"], warnings, None
    ratio = len(target.strip()) / max(len(source.strip()), 1)
    if not config["min_length_ratio"] <= ratio <= config["max_length_ratio"]:
        warnings.append("length_ratio_outside_range")
    if ratio < config["truncation_ratio"] or (len(source) > 100 and target.rstrip().endswith(("...", "…")) and not source.rstrip().endswith(("...", "…"))):
        warnings.append("possible_truncation")
    if source.strip().casefold() == target.strip().casefold():
        warnings.append("unchanged_text")
    protected = {value.rstrip(".,;:!?)\"'") for value in URL_PATTERN.findall(source)}
    protected.update(term for term in config["protected_terms"] if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", source, re.IGNORECASE))
    missing = sorted(term for term in protected if not re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", target, re.IGNORECASE))
    if missing:
        warnings.append("protected_terms_missing: " + ", ".join(missing))
    return errors, warnings, ratio


def review_sample(frame: pd.DataFrame, labels: list[str], config: dict, seed: int) -> pd.DataFrame:
    """Amostra reprodutível que cobre classes e splits, além de casos sinalizados."""
    rng = random.Random(seed)
    ids = sorted(frame.segment_id.tolist())
    selected: set[str] = set()
    for label in labels:
        pool = sorted(frame.loc[frame[label] == 1, "segment_id"])
        selected.update(rng.sample(pool, min(int(config["sample_per_class"]), len(pool))))
    for _, rows in frame.groupby("split", sort=True):
        selected.add(rng.choice(sorted(rows.segment_id)))
    flagged = sorted(frame.loc[(frame.validation_errors != "") | (frame.validation_warnings != ""), "segment_id"])
    selected.update(rng.sample(flagged, min(len(flagged), int(config["sample_size"]) // 2)))
    remaining = sorted(set(ids) - selected)
    selected.update(rng.sample(remaining, min(len(remaining), max(0, int(config["sample_size"]) - len(selected)))))
    columns = ["segment_id", "policy_id", "split", "labels", "texto_en", "texto_pt", "validation_errors", "validation_warnings", "length_ratio"]
    sample = frame.loc[frame.segment_id.isin(selected), columns].sort_values("segment_id").copy()
    for column in ["reviewer", "meaning_preserved", "negation_preserved", "scope_preserved", "review_notes"]:
        sample[column] = ""
    return sample


def validate_translation(frame: pd.DataFrame, types: dict[str, str], config: dict) -> tuple[pd.DataFrame, dict]:
    """Produz relatório de todos os trechos; avisos nunca equivalem a aprovação humana."""
    result = frame.copy()
    checks = [translation_issues(row.texto_en, row.texto_pt, types.get(row.segment_id, "s"), config)
              for row in frame.itertuples()]
    result["validation_errors"] = [" | ".join(check[0]) for check in checks]
    result["validation_warnings"] = [" | ".join(check[1]) for check in checks]
    result["length_ratio"] = [check[2] for check in checks]
    counts = Counter(issue.split(":", 1)[0] for errors, warnings, _ in checks for issue in errors + warnings)
    return result, {"segments": len(result), "rows_with_errors": int((result.validation_errors != "").sum()),
                    "rows_with_warnings": int((result.validation_warnings != "").sum()),
                    "issues": dict(counts), "human_review_status": "pending",
                    "automatic_checks_passed": not any(errors for errors, _, _ in checks)}
