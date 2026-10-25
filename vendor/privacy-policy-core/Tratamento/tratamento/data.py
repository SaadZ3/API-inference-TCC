"""Leitura de segmentos e votos sem depender dos índices de spans."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd


class TextExtractor(HTMLParser):
    """Remove marcação preservando separação entre blocos e texto inline."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"br", "p", "div", "li", "h1", "h2", "h3", "tr", "td"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "div", "li", "h1", "h2", "h3", "tr", "td"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def clean_html(raw: str) -> str:
    """Normaliza espaços após interpretar entidades HTML uma única vez."""
    parser = TextExtractor()
    parser.feed(raw)
    parser.close()
    return " ".join("".join(parser.parts).split())


def text_hash(text: str) -> str:
    """Identifica textos iguais, ignorando caixa e espaços redundantes."""
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def load_categories(root: Path) -> list[str]:
    """Obtém a ordem oficial diretamente do esquema XML local."""
    tree = ElementTree.parse(root / "documentation/categories-july30.xml")
    labels = [str(node.text) for node in tree.findall("{*}category/{*}name")]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Esquema de categorias vazio ou com duplicatas.")
    return labels


def load_corpus(root: Path, min_votes: int = 1) -> tuple[pd.DataFrame, list[str], dict]:
    """Une categorias por segmento contando anotadores, não práticas repetidas."""
    if min_votes not in {1, 2, 3}:
        raise ValueError("min_votes deve ser 1, 2 ou 3.")
    labels = load_categories(root)
    files = sorted((root / "annotations").glob("*.csv"))
    if not files:
        raise ValueError("Nenhum CSV encontrado em annotations/.")
    records: list[dict[str, Any]] = []
    seen_annotations: set[str] = set()
    seen_policies: set[str] = set()
    annotation_count = 0
    mismatches = 0
    for path in files:
        policy_id = path.stem.split("_", 1)[0]
        if policy_id in seen_policies:
            raise ValueError(f"policy_id duplicado nos nomes: {policy_id}")
        seen_policies.add(policy_id)
        source = root / "sanitized_policies" / f"{path.stem}.html"
        segments = source.read_text(encoding="utf-8-sig").split("|||")
        votes: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        internal_ids: set[str] = set()
        annotators: set[str] = set()
        urls: set[str] = set()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line, row in enumerate(csv.reader(handle), 1):
                if len(row) != 9:
                    raise ValueError(f"{path}:{line}: esperadas 9 colunas, recebidas {len(row)}.")
                annotation_id, _, annotator, internal_id, index, label = row[:6]
                if annotation_id in seen_annotations:
                    raise ValueError(f"ID de anotação duplicado: {annotation_id}")
                seen_annotations.add(annotation_id)
                index = int(index)
                if not 0 <= index < len(segments):
                    raise ValueError(f"{path}:{line}: segmento fora do intervalo: {index}")
                if label not in labels:
                    raise ValueError(f"Categoria desconhecida: {label}")
                json.loads(row[6])
                votes[index][label].add(annotator)
                internal_ids.add(internal_id)
                annotators.add(annotator)
                # O arquivo real tem data e URL em ordem diferente do manual.
                urls.update(value for value in row[7:] if value.startswith(("http://", "https://")))
                annotation_count += 1
        if len(internal_ids) != 1 or len(annotators) != 3:
            raise ValueError(f"{path}: esperado um ID interno e três anotadores por política.")
        mismatches += policy_id not in internal_ids
        for index, raw in enumerate(segments):
            text = clean_html(raw)
            if not text:
                raise ValueError(f"{source}: segmento vazio {index}; índices não foram renumerados.")
            counts = {label: len(votes[index].get(label, set())) for label in labels}
            selected = [label for label in labels if counts[label] >= min_votes]
            record: dict[str, Any] = {
                "segment_id": f"{policy_id}:{index}", "policy_id": policy_id,
                "texto_en": text, "labels": json.dumps(selected, ensure_ascii=False),
            }
            record.update({label: int(label in selected) for label in labels})
            record.update({
                "n_labels": len(selected), "split": "", "texto_pt": "",
                "segment_index": index, "annotation_policy_id": next(iter(internal_ids)),
                "source_file": source.relative_to(root).as_posix(),
                "annotation_file": path.relative_to(root).as_posix(),
                "policy_url": " | ".join(sorted(urls)),
                "text_sha256": text_hash(text),
                "label_votes": json.dumps(counts, ensure_ascii=False),
                "n_labels_union": sum(n >= 1 for n in counts.values()),
                "n_labels_majority": sum(n >= 2 for n in counts.values()),
            })
            records.append(record)
    frame = pd.DataFrame(records)
    return frame, labels, {"annotations": annotation_count, "policies": len(files),
                           "policy_id_mismatches": mismatches, "min_votes": min_votes}
