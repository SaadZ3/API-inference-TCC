"""Trechos rastreáveis, limitados pelo tokenizer exportado com o classificador."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from .extraction import ExtractedDocument, IngestionError

PREPROCESSING_VERSION = "p3-v3"


class PolicySegmenter:
    """Carrega somente tokenizer/configuração locais; nenhum peso ou download."""

    def __init__(self, model_dir: str | Path) -> None:
        from transformers import AutoTokenizer

        directory = Path(model_dir)
        try:
            settings = json.loads((directory / "inference.json").read_text(encoding="utf-8"))
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            max_length = settings["max_length"]
            if type(max_length) is not int or not 2 < max_length <= config["max_position_embeddings"]:
                raise ValueError("Limite de tokens inválido.")
            verification = directory / "export_verification.json"
            if verification.exists() and json.loads(verification.read_text(encoding="utf-8"))["status"] != "verified":
                raise ValueError("Exportação não verificada.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                directory, use_fast=True, local_files_only=True, trust_remote_code=False
            )
            if not self.tokenizer.is_fast:
                raise ValueError("A segmentação exige offsets do tokenizer rápido.")
            self.max_length = max_length
            self.payload_tokens = max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
            if self.payload_tokens < 1:
                raise ValueError("Limite insuficiente para os tokens especiais.")
            fingerprint = sha256()
            for name in ["inference.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]:
                fingerprint.update(name.encode("utf-8"))
                fingerprint.update((directory / name).read_bytes())
            self.tokenizer_sha256 = fingerprint.hexdigest()
        except (OSError, ValueError, KeyError, TypeError):
            raise IngestionError("INFERENCE_UNAVAILABLE", "Não foi possível carregar o tokenizer/configuração do pacote local.") from None

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])

    def _chunks(self, text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
        """Recorta caracteres originais; nunca reconstrói o texto com decode(tokens)."""
        while start < end:
            while start < end and text[start].isspace():
                start += 1
            if start == end:
                return
            remainder = text[start:end]
            encoded = self.tokenizer(
                remainder, add_special_tokens=False, truncation=False,
                return_offsets_mapping=True, verbose=False,
            )
            if len(encoded["input_ids"]) <= self.payload_tokens:
                cut = end
            else:
                offsets = encoded["offset_mapping"]
                cut = start + offsets[self.payload_tokens - 1][1]
                if cut <= start:
                    raise IngestionError("UNREADABLE_FILE", "Não foi possível segmentar o texto com os offsets do tokenizer.")
                # Prefere fim de frase; depois espaço; em último caso divide dentro da palavra.
                prefix = text[start:cut]
                sentences = list(re.finditer(r'[.!?][\"\u201d\u2019)]*\s+', prefix))
                spaces = list(re.finditer(r"\s+", prefix))
                word_boundary = len(prefix) if text[cut:cut + 1].isspace() else (spaces[-1].end() if spaces else None)
                boundary = sentences[-1].end() if sentences else word_boundary
                if boundary:
                    cut = start + boundary
            while cut > start and text[cut - 1].isspace():
                cut -= 1
            # O prefixo isolado pode tokenizar diferente do parágrafo completo.
            while cut > start and self.count_tokens(text[start:cut]) > self.max_length:
                cut -= 1
            if cut <= start:
                raise IngestionError("UNREADABLE_FILE", "Não foi possível produzir um trecho dentro do limite do modelo.")
            yield start, cut
            start = cut

    def prepare(self, document: ExtractedDocument) -> dict:
        warnings = list(document.warnings)
        partial = document.status == "partial"
        segments = []
        for page in document.pages:
            page_text = document.text[page.char_start:page.char_end]
            # Linhas em branco delimitam parágrafos; quebras simples de PDF são preservadas.
            for paragraph in re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", page_text):
                start = page.char_start + paragraph.start()
                end = page.char_start + paragraph.end()
                while end > start and document.text[end - 1].isspace():
                    end -= 1
                for left, right in self._chunks(document.text, start, end):
                    text = document.text[left:right]
                    n_tokens = self.count_tokens(text)
                    if n_tokens <= self.tokenizer.num_special_tokens_to_add(pair=False):
                        partial = True
                        warnings.append(f"UNUSABLE_TEXT: intervalo {left}:{right} não produz tokens de conteúdo.")
                        continue
                    segments.append({
                        "segment_id": f"{document.sha256[:16]}:{left}:{right}",
                        "text": text, "page_start": page.page_number, "page_end": page.page_number,
                        "char_start": left, "char_end": right,
                        "n_tokens": n_tokens, "truncated": False,
                    })
        if not segments:
            raise IngestionError("UNREADABLE_FILE", "Nenhum trecho utilizável pelo tokenizer foi encontrado.")
        return {
            "schema_version": 1,
            "language": "pt",
            "document_scope": document.document_scope,
            "source": {
                "filename": document.filename, "sha256": document.sha256,
                "format": document.format, "byte_size": document.byte_size,
            },
            "preprocessing": {
                "version": PREPROCESSING_VERSION,
                "max_length": self.max_length,
                "tokenizer_sha256": self.tokenizer_sha256,
            },
            "extraction": {
                "status": "partial" if partial else "ok",
                "warnings": warnings,
                "text": document.text,
                "page_count": len(document.pages) if document.format == "pdf" else None,
                "pages": [asdict(page) for page in document.pages],
            },
            "segments": segments,
        }
