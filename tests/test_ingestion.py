from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
from pypdf import PdfWriter
from pypdf._page import PageObject
from pypdf.generic import DictionaryObject, NameObject, NumberObject, DecodedStreamObject

from privacy_service.extraction import ExtractionLimits, IngestionError, extract_document, extract_path
from privacy_service.segmentation import PolicySegmenter


def pdf_bytes(pages, *, encrypted=False, image_pages=()):
    """PDFs pequenos e determinísticos para cenários que não dependem de rede."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"), NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
    })
    font_ref = writer._add_object(font)
    for number, text in enumerate(pages, 1):
        page = writer.add_blank_page(width=595, height=842)
        resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
        commands = b"BT /F1 12 Tf 40 780 Td 18 TL "
        for line in text.split("\n"):
            commands += b"<" + line.encode("cp1252").hex().encode("ascii") + b"> Tj T* "
        commands += b"ET"
        if number in image_pages:
            image = DecodedStreamObject()
            image.set_data(b"\xff\xff\xff")
            image.update({
                NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"),
                NameObject("/Width"): NumberObject(1), NameObject("/Height"): NumberObject(1),
                NameObject("/ColorSpace"): NameObject("/DeviceRGB"), NameObject("/BitsPerComponent"): NumberObject(8),
            })
            resources[NameObject("/XObject")] = DictionaryObject({NameObject("/Im0"): writer._add_object(image)})
            commands += b" q 100 0 0 100 40 500 cm /Im0 Do Q"
        page[NameObject("/Resources")] = resources
        stream = DecodedStreamObject()
        stream.set_data(commands)
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("senha-teste")
    result = BytesIO()
    writer.write(result)
    return result.getvalue()


def extract(content, name="politica.txt", **kwargs):
    return extract_document(content, name, document_scope="full_document", **kwargs)


def assert_preserved(result, segmenter):
    source = result["extraction"]["text"]
    last = 0
    for segment in result["segments"]:
        left, right = segment["char_start"], segment["char_end"]
        assert left >= last
        assert not source[last:left].strip()
        assert source[left:right] == segment["text"]
        assert segment["n_tokens"] == segmenter.count_tokens(segment["text"]) <= 384
        assert segment["truncated"] is False
        last = right
    assert not source[last:].strip()
    assert len({item["segment_id"] for item in result["segments"]}) == len(result["segments"])


def test_txt_bom_accents_line_endings_and_hash(segmenter):
    raw = "\ufeff  Política de proteção de dados.\r\n\r\nNão vendemos informações.\rContato: ação@example.test.  ".encode("utf-8")
    doc = extract(raw, "C:\\pasta\\politica.TXT")
    assert doc.filename == "politica.TXT"
    assert "\r" not in doc.text and "\ufeff" not in doc.text
    assert "Não" in doc.text
    result = segmenter.prepare(doc)
    assert result["extraction"]["status"] == "ok"
    assert result["extraction"]["page_count"] is None
    assert all(segment["page_start"] is None for segment in result["segments"])
    assert_preserved(result, segmenter)
    assert result == segmenter.prepare(extract(raw, "C:\\pasta\\politica.TXT"))


@pytest.mark.parametrize("scope", ["full_document", "excerpt", "unknown"])
def test_scope_is_not_inferred_from_extraction(scope, segmenter):
    doc = extract_document(b"Dados pessoais.", "p.txt", document_scope=scope)
    result = segmenter.prepare(doc)
    assert result["document_scope"] == scope
    assert result["extraction"]["status"] == "ok"
    assert "score" not in result and "evaluation" not in result


@pytest.mark.parametrize("content,name,code", [
    (b"", "p.txt", "UNREADABLE_FILE"), (b" \n\t", "p.txt", "UNREADABLE_FILE"),
    (b"\xff\xfe", "p.txt", "UNREADABLE_FILE"), (b"a\x00b", "p.txt", "UNREADABLE_FILE"),
    (b"texto", "p.docx", "UNSUPPORTED_FORMAT"), (b"texto", "p.pdf", "UNREADABLE_FILE"),
    (b"%PDF-1.7\nincompleto", "p.pdf", "UNREADABLE_FILE"),
    (b"%PDF-1.7", "p.txt", "UNREADABLE_FILE"),
    ("Texto \ufffd".encode(), "p.txt", "UNREADABLE_FILE"),
])
def test_invalid_files(content, name, code):
    with pytest.raises(IngestionError) as failure:
        extract(content, name)
    assert failure.value.code == code


def test_invalid_scope():
    with pytest.raises(IngestionError, match="escopo"):
        extract_document(b"Texto", "p.txt", document_scope="complete")


def test_pdf_multiple_pages_and_accents(segmenter):
    doc = extract(pdf_bytes(["Política de privacidade. Não vendemos dados.", "Exclusão e correção. Contato: privacidade@example.test."]), "p.pdf")
    result = segmenter.prepare(doc)
    assert result["extraction"]["page_count"] == 2
    assert result["extraction"]["status"] == "ok"
    assert "Exclusão" in doc.text
    assert {s["page_start"] for s in result["segments"]} == {1, 2}
    for segment in result["segments"]:
        page = result["extraction"]["pages"][segment["page_start"] - 1]
        assert page["char_start"] <= segment["char_start"] < segment["char_end"] <= page["char_end"]
    assert_preserved(result, segmenter)


@pytest.mark.parametrize("image", [False, True])
def test_pdf_missing_page_is_partial(image, segmenter):
    raw = pdf_bytes(["Dados da conta.", "", "Seus direitos."], image_pages=[2] if image else [])
    doc = extract(raw, "p.pdf")
    result = segmenter.prepare(doc)
    assert result["extraction"]["status"] == "partial"
    assert result["extraction"]["pages"][1]["status"] == "empty"
    assert any("página 2" in w for w in result["extraction"]["warnings"])
    assert {s["page_start"] for s in result["segments"]} == {1, 3}
    assert result["document_scope"] == "full_document"
    assert_preserved(result, segmenter)


@pytest.mark.parametrize("pages,images,encrypted,code", [
    ([""], [], False, "UNREADABLE_FILE"),
    ([], [], False, "UNREADABLE_FILE"),
    ([""], [1], False, "OCR_REQUIRED"),
    (["Texto"], [], True, "UNREADABLE_FILE"),
])
def test_pdf_unusable(pages, images, encrypted, code):
    with pytest.raises(IngestionError) as failure:
        extract(pdf_bytes(pages, image_pages=images, encrypted=encrypted), "p.pdf")
    assert failure.value.code == code


def test_page_extraction_exception_remains_visible(monkeypatch, segmenter):
    original = PageObject.extract_text
    calls = 0
    def broken(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Falha simulada da biblioteca")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PageObject, "extract_text", broken)
    result = segmenter.prepare(extract(pdf_bytes(["Primeira.", "Segunda.", "Terceira."]), "p.pdf"))
    assert result["extraction"]["status"] == "partial"
    assert result["extraction"]["pages"][1]["status"] == "failed"
    assert any("PAGE_EXTRACTION_FAILED" in w for w in result["extraction"]["warnings"])
    assert {s["page_start"] for s in result["segments"]} == {1, 3}


@pytest.mark.parametrize("field,content,name", [
    ("max_bytes", b"Dados pessoais", "p.txt"),
    ("max_characters", "Proteção de dados".encode(), "p.txt"),
    ("max_pages", pdf_bytes(["A", "B"]), "p.pdf"),
    ("max_page_stream_bytes", pdf_bytes(["Texto da política"]), "p.pdf"),
])
def test_limits_are_errors_not_truncation(field, content, name):
    limits = replace(ExtractionLimits(), **{field: 1})
    with pytest.raises(IngestionError) as failure:
        extract(content, name, limits=limits)
    assert failure.value.code == "FILE_TOO_LARGE"


def test_path_bounded_read(tmp_path):
    path = tmp_path / "p.txt"
    path.write_bytes(b"x" * 100)
    with pytest.raises(IngestionError) as failure:
        extract_path(path, document_scope="unknown", limits=ExtractionLimits(max_bytes=10))
    assert failure.value.code == "FILE_TOO_LARGE"


@pytest.mark.parametrize("text", [
    ("Coletamos informações para criar sua conta. Você pode solicitar a exclusão dos seus dados. " * 200),
    ("proteção dados pessoais consentimento armazenamento " * 500),
    ("segurança/privacidade;" * 600),
    ("  Ação, saúde, São Paulo!\n\n" * 100),
    ("e\u0301 🛡️ ＡＢＣ 数据 [CLS] [SEP] " * 200),
])
def test_long_text_preserved_without_truncation(text, segmenter):
    result = segmenter.prepare(extract(text.encode()))
    assert len(result["segments"]) > 1
    assert_preserved(result, segmenter)


def test_exact_token_limit_includes_special_tokens(segmenter):
    for amount in [381, 382, 383, 764]:
        text = " ".join(["dados"] * amount)
        assert segmenter.count_tokens(text) == amount + 2
        result = segmenter.prepare(extract(text.encode()))
        assert len(result["segments"]) == (amount + 381) // 382
        assert_preserved(result, segmenter)


def test_seeded_varied_whitespace_and_punctuation(segmenter):
    rng = random.Random(37)
    words = ["privacidade", "João", "dados", "Não!", "123", "direitos.", "proteção?", "👩🏽‍💻"]
    for _ in range(15):
        text = "".join(rng.choice(words) + rng.choice([" ", "\t", "\n", "\n\n", "  "]) for _ in range(700))
        assert_preserved(segmenter.prepare(extract(text.encode())), segmenter)


def test_no_content_tokens_is_error(segmenter):
    with pytest.raises(IngestionError) as failure:
        segmenter.prepare(extract("\u200b\u200b".encode()))
    assert failure.value.code == "UNREADABLE_FILE"


def test_invalid_model_is_local_error(tmp_path):
    with pytest.raises(IngestionError) as failure:
        PolicySegmenter(tmp_path)
    assert failure.value.code == "INFERENCE_UNAVAILABLE"


def test_cli_writes_json_and_refuses_overwrite(tmp_path, model_dir):
    input_file = tmp_path / "política com espaços.txt"
    output = tmp_path / "preparada.json"
    input_file.write_text("Não vendemos dados. Você pode solicitar exclusão.", encoding="utf-8")
    args = [sys.executable, "-m", "privacy_service.prepare", "--input", str(input_file),
            "--output", str(output), "--model-dir", str(model_dir), "--document-scope", "excerpt"]
    first = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    assert first.returncode == 0, first.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["document_scope"] == "excerpt"
    assert len(result["segments"]) == 1
    before = output.read_bytes()
    again = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    assert again.returncode != 0
    assert output.read_bytes() == before
    args[args.index("--output") + 1] = str(input_file)
    same = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    assert same.returncode != 0
    assert input_file.read_text(encoding="utf-8").startswith("Não vendemos")


def test_cli_invalid_file_creates_no_output(tmp_path, model_dir):
    path = tmp_path / "invalido.pdf"
    path.write_bytes(b"not PDF")
    output = tmp_path / "out.json"
    run = subprocess.run([sys.executable, "-m", "privacy_service.prepare", "--input", str(path),
                          "--output", str(output), "--model-dir", str(model_dir),
                          "--document-scope", "unknown"], capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 1
    assert json.loads(run.stderr)["error"]["code"] == "UNREADABLE_FILE"
    assert not output.exists()

def test_pdf_parser_warning_marks_partial(monkeypatch, segmenter):
    import logging
    original = PageObject.extract_text
    def with_warning(self, *args, **kwargs):
        logging.getLogger("pypdf._page").warning("Aviso simulado do parser")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PageObject, "extract_text", with_warning)
    result = segmenter.prepare(extract(pdf_bytes(["Texto legível."]), "p.pdf"))
    assert result["extraction"]["status"] == "partial"
    assert any("PDF_PARSER_WARNING" in w for w in result["extraction"]["warnings"])
    assert result["segments"][0]["text"] == "Texto legível."
    assert not any(type(handler).__name__ == "_PdfWarnings" for handler in logging.getLogger("pypdf").handlers)


def test_unusable_paragraph_does_not_hide_partial_status(segmenter):
    result = segmenter.prepare(extract("Texto válido.\n\n\u200b\u200b".encode()))
    assert result["extraction"]["status"] == "partial"
    assert any("UNUSABLE_TEXT" in w for w in result["extraction"]["warnings"])
    assert "\u200b" in result["extraction"]["text"]

@pytest.mark.parametrize("max_length", [True, 2, 513])
def test_invalid_package_token_limit_is_rejected(tmp_path, model_dir, max_length):
    import shutil
    for name in ["inference.json", "config.json"]:
        shutil.copy2(model_dir / name, tmp_path / name)
    settings = json.loads((tmp_path / "inference.json").read_text(encoding="utf-8"))
    settings["max_length"] = max_length
    (tmp_path / "inference.json").write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(IngestionError) as failure:
        PolicySegmenter(tmp_path)
    assert failure.value.code == "INFERENCE_UNAVAILABLE"


def test_reads_token_limit_from_package(tmp_path, model_dir):
    import shutil
    for name in ["inference.json", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]:
        shutil.copy2(model_dir / name, tmp_path / name)
    settings = json.loads((tmp_path / "inference.json").read_text(encoding="utf-8"))
    settings["max_length"] = 20
    (tmp_path / "inference.json").write_text(json.dumps(settings), encoding="utf-8")
    small = PolicySegmenter(tmp_path)
    result = small.prepare(extract(("dados " * 30).encode()))
    assert result["preprocessing"]["max_length"] == 20
    assert len(result["segments"]) == 2
    assert all(s["n_tokens"] <= 20 for s in result["segments"])
