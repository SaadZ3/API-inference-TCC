"""Extração local de PDF digital e TXT, com perdas de leitura sinalizadas e sem OCR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import logging
import threading
from typing import Any
import unicodedata

from pypdf import PdfReader
import pdfplumber

from .pdf_reading_order import read_positioned_page

SCOPES = {"full_document", "excerpt", "unknown"}


class IngestionError(ValueError):
    """Erro público, sem conteúdo do documento ou caminhos internos."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ExtractionLimits:
    max_bytes: int = 10 * 1024 * 1024
    max_pages: int = 50
    max_characters: int = 100_000
    max_page_stream_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Os limites devem ser inteiros positivos.")


@dataclass(frozen=True)
class PageSpan:
    page_number: int | None
    char_start: int
    char_end: int
    status: str


@dataclass
class ExtractedDocument:
    filename: str
    sha256: str
    format: str
    byte_size: int
    document_scope: str
    text: str
    pages: list[PageSpan]
    warnings: list[str]
    status: str


def _normalize(text: str) -> str:
    # Mantém caixa, acentos, hifens, cabeçalhos e espaços. Offsets referem-se a esta versão.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _suspicious_characters(text: str) -> bool:
    return "\ufffd" in text or any(
        unicodedata.category(char) == "Cc" and char not in "\n\t\f" for char in text
    )


def _has_image(resources: Any, visited: set[int] | None = None) -> bool:
    """Reconhece imagens diretas e em formulários, sem decodificar seus pixels."""
    visited = set() if visited is None else visited
    if resources is None or len(visited) >= 100:
        return False
    resources = resources.get_object()
    if id(resources) in visited:
        return False
    visited.add(id(resources))
    objects = resources["/XObject"].get_object() if "/XObject" in resources else {}
    for reference in objects.values():
        obj = reference.get_object()
        if obj.get("/Subtype") == "/Image":
            return True
        if obj.get("/Subtype") == "/Form" and _has_image(obj.get("/Resources"), visited):
            return True
    return False


class _PdfWarnings(logging.Handler):
    """Registra a existência de avisos, sem copiar detalhes do PDF para o JSON."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.thread_id = threading.get_ident()
        self.seen = False
        self.xref_corrected = False

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread_id:
            # O pypdf corrige esta numeração ao ler PDFs válidos, inclusive
            # arquivos otimizados. Outros avisos continuam impeditivos.
            if (record.name == 'pypdf._reader' and record.levelno == logging.WARNING
                    and record.getMessage() == 'Xref table not zero-indexed. ID numbers for objects will be corrected.'):
                self.xref_corrected = True
            else:
                self.seen = True

    def __enter__(self):
        logging.getLogger("pypdf").addHandler(self)
        return self

    def __exit__(self, *_args) -> None:
        logging.getLogger("pypdf").removeHandler(self)
        self.close()


def extract_document(
    content: bytes,
    filename: str,
    *,
    document_scope: str,
    limits: ExtractionLimits | None = None,
) -> ExtractedDocument:
    """Extrai bytes de upload; não recebe URL nem abre caminhos fornecidos por clientes."""
    limits = limits or ExtractionLimits()
    if document_scope not in SCOPES:
        raise IngestionError("INVALID_INPUT", "Informe o escopo: full_document, excerpt ou unknown.")
    if not isinstance(content, bytes):
        raise IngestionError("INVALID_INPUT", "O conteúdo deve ser fornecido em bytes.")
    if not isinstance(filename, str) or not filename.strip():
        raise IngestionError("INVALID_INPUT", "Informe o nome do arquivo.")
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    kind = Path(filename).suffix.lower()
    if kind not in {".txt", ".pdf"}:
        raise IngestionError("UNSUPPORTED_FORMAT", "Envie PDF com texto ou TXT em UTF-8.")
    if len(content) > limits.max_bytes:
        raise IngestionError("FILE_TOO_LARGE", f"O arquivo excede {limits.max_bytes} bytes.")
    if not content:
        raise IngestionError("UNREADABLE_FILE", "O arquivo está vazio.")

    warnings: list[str] = []
    parser_incomplete = False
    extracted: list[tuple[int | None, str, str]] = []
    if kind == ".txt":
        try:
            text = _normalize(content.decode("utf-8-sig"))
        except UnicodeDecodeError:
            raise IngestionError("UNREADABLE_FILE", "O TXT deve estar codificado em UTF-8.") from None
        if content.lstrip().startswith(b"%PDF-") or _suspicious_characters(text):
            raise IngestionError("UNREADABLE_FILE", "O TXT contém dados binários ou caracteres inválidos.")
        extracted.append((None, text, "ok" if text.strip() else "empty"))
    else:
        if not content.lstrip().startswith(b"%PDF-"):
            raise IngestionError("UNREADABLE_FILE", "O arquivo não possui uma assinatura PDF válida.")
        try:
            with _PdfWarnings() as notices:
                reader = PdfReader(BytesIO(content), strict=True)
                if reader.is_encrypted:
                    raise IngestionError("UNREADABLE_FILE", "PDF protegido por senha não é suportado.")
                count = len(reader.pages)
                if count > limits.max_pages:
                    raise IngestionError("FILE_TOO_LARGE", f"O PDF excede {limits.max_pages} páginas.")
                if count == 0:
                    raise IngestionError("UNREADABLE_FILE", "O PDF não possui páginas.")
        except IngestionError:
            raise
        except Exception:
            raise IngestionError("UNREADABLE_FILE", "Não foi possível abrir o PDF. Verifique se ele está íntegro.") from None

        if notices.seen:
            parser_incomplete = True
            warnings.append("PDF_PARSER_WARNING: a biblioteca informou inconsistências ao abrir o PDF; revise a extração.")
        if notices.xref_corrected:
            warnings.append("PDF_XREF_INDEX_CORRECTED: a numeração interna do PDF foi ajustada automaticamente; o aviso isolado não indica perda de texto.")
        images_found = False
        failed = False
        characters = 0
        with ExitStack() as resources:
            positioned_pdf = None
            for index in range(count):
                number = index + 1
                try:
                    with _PdfWarnings() as notices:
                        page = reader.pages[index]
                        images_found = _has_image(page.get("/Resources")) or images_found
                        stream = page.get_contents()
                        if stream is not None and len(stream.get_data()) > limits.max_page_stream_bytes:
                            raise IngestionError("FILE_TOO_LARGE", f"A página {number} excede o limite de conteúdo PDF descompactado.")
                        text = _normalize(page.extract_text() or "")
                        if text.strip() and not _suspicious_characters(text):
                            try:
                                if positioned_pdf is None:
                                    positioned_pdf = resources.enter_context(pdfplumber.open(BytesIO(content)))
                                text = read_positioned_page(positioned_pdf.pages[index], text)
                            except Exception:
                                warnings.append(f"PDF_READING_ORDER_FALLBACK: página {number}: mantida a leitura original; confira a posição de links e tabelas.")
                except IngestionError:
                    raise
                except Exception:
                    extracted.append((number, "", "failed"))
                    warnings.append(f"PAGE_EXTRACTION_FAILED: não foi possível extrair a página {number}.")
                    failed = True
                    continue
                null_count = text.count("\x00")
                if null_count:
                    # JSONB não aceita NUL. O marcador mantém a posição e torna a perda
                    # visível, sem inventar qual caractere estava no PDF.
                    text = text.replace("\x00", "\ufffd")
                    warnings.append(
                        f"PDF_NULL_CHARACTERS_REPLACED: página {number}: {null_count} caracteres "
                        "não puderam ser lidos e foram marcados com �; confira o PDF original."
                    )
                characters += len(text)
                if characters > limits.max_characters:
                    raise IngestionError("FILE_TOO_LARGE", f"O texto excede {limits.max_characters} caracteres.")
                status = "ok"
                if notices.seen:
                    status = "partial"
                    warnings.append(f"PDF_PARSER_WARNING: a biblioteca informou inconsistências na página {number}; revise a extração.")
                if notices.xref_corrected:
                    warnings.append(f"PDF_XREF_INDEX_CORRECTED: a numeração interna foi ajustada na página {number}; o aviso isolado não indica perda de texto.")
                if not text.strip():
                    status = "empty"
                    warnings.append(f"PAGE_WITHOUT_TEXT: página {number} sem texto extraível; pode estar vazia ou exigir OCR.")
                elif _suspicious_characters(text):
                    status = "partial"
                    warnings.append(f"SUSPICIOUS_CHARACTERS: página {number} contém caracteres que exigem revisão.")
                extracted.append((number, text, status))
        if not any(text.strip() for _, text, _ in extracted):
            if images_found and not failed:
                raise IngestionError("OCR_REQUIRED", "O PDF contém imagens, mas nenhum texto extraível. É necessário OCR, fora do protótipo.")
            raise IngestionError("UNREADABLE_FILE", "Nenhum texto utilizável foi extraído do PDF.")

    parts: list[str] = []
    spans: list[PageSpan] = []
    offset = 0
    for number, text, status in extracted:
        if parts:
            parts.append("\n\n")
            offset += 2
        spans.append(PageSpan(number, offset, offset + len(text), status))
        parts.append(text)
        offset += len(text)
    text = "".join(parts)
    if len(text) > limits.max_characters:
        raise IngestionError("FILE_TOO_LARGE", f"O texto excede {limits.max_characters} caracteres.")
    if not text.strip():
        raise IngestionError("UNREADABLE_FILE", "O documento não contém texto utilizável.")
    return ExtractedDocument(
        filename, sha256(content).hexdigest(), kind[1:], len(content), document_scope,
        text, spans, warnings, "partial" if parser_incomplete or any(p.status != 'ok' for p in spans) else "ok",
    )


def extract_path(
    path: str | Path, *, document_scope: str, limits: ExtractionLimits | None = None
) -> ExtractedDocument:
    """Adaptador apenas da CLI; lê no máximo limite + 1 byte."""
    limits = limits or ExtractionLimits()
    path = Path(path)
    try:
        with path.open("rb") as stream:
            content = stream.read(limits.max_bytes + 1)
    except OSError:
        raise IngestionError("UNREADABLE_FILE", "Não foi possível ler o arquivo de entrada.") from None
    return extract_document(content, path.name, document_scope=document_scope, limits=limits)
