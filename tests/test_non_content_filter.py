"""Títulos e datas não viram evidências, sem descartar declarações curtas."""
import re
import time
from types import SimpleNamespace

import pytest

from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.scoring_common import LABELS
from privacy_service.extraction import IngestionError, extract_document
from privacy_service.pipeline import AnalysisPipeline
from privacy_service.segmentation import PREPROCESSING_VERSION, non_content_reason
from privacy_service.settings import Settings


@pytest.mark.parametrize("text,reason", [
    ("Alterações nesta política", "título sem declaração"),
    ("Alterações na política", "título sem declaração"),
    ("12. Alterações da política de privacidade", "título sem declaração"),
    ("“Alterações nesta política”", "título sem declaração"),
    ("ANEXO I 1. POLÍTICA DE PRIVACIDADE DE ALUNOS", "título sem declaração"),
    ("ANEXO I\n1. POLÍTICA DE PRIVACIDADE DE ALUNOS", "título sem declaração"),
    ("Segurança de dados", "título sem declaração"),
    ("Direitos dos titulares", "título sem declaração"),
    ("Outubro /2026", "data isolada"),
    ("“Outubro /2026”", "data isolada"),
    ("outubro de 2026", "data isolada"),
    ("Última atualização: 10 de outubro de 2026.", "data isolada"),
    ("16/11/2026", "data isolada"),
])
def test_recognizes_only_standalone_headings_and_dates(text, reason):
    assert non_content_reason(text) == reason


@pytest.mark.parametrize("text", [
    "Não compartilhamos seus dados.",
    "NÃO VENDEMOS DADOS.",
    "Coletamos nome e e-mail.",
    "Guardamos dados por 30 dias.",
    "Dados: nome, e-mail.",
    "Contato: privacidade@example.test",
    "Segurança de dados: criptografia.",
    "Alterações nesta política serão avisadas por e-mail.",
    "Alterações nesta política\nAvisaremos por e-mail.",
    "Outubro /2026\nColetamos dados pessoais.",
    "Desde outubro de 2026 não coletamos dados de crianças.",
    "ANEXO I 1. POLÍTICA DE PRIVACIDADE DE ALUNOS\nNão coletamos dados de menores.",
    "Para alunos menores, exigimos autorização dos responsáveis.",
    "Crianças menores de 13 anos.",
    "Nome e e-mail",
])
def test_preserves_short_statements_and_mixed_paragraphs(text):
    assert non_content_reason(text) is None


def prepare(segmenter, text):
    return segmenter.prepare(extract_document(
        text.encode("utf-8"), "politica.txt", document_scope="full_document"))


def test_filter_keeps_original_positions_and_audit_without_marking_partial(segmenter):
    text = ("Alterações nesta política\n\nNão compartilhamos seus dados.\n\n"
            "Outubro /2026\n\nANEXO I 1. POLÍTICA DE PRIVACIDADE DE ALUNOS\n\n"
            "Para alunos menores, exigimos autorização dos responsáveis.")
    result = prepare(segmenter, text)
    assert result == prepare(segmenter, text)
    assert result["preprocessing"]["version"] == PREPROCESSING_VERSION == "p3-v4"
    assert result["extraction"]["text"] == text
    assert result["extraction"]["status"] == "ok"
    assert [s["text"] for s in result["segments"]] == [
        "Não compartilhamos seus dados.",
        "Para alunos menores, exigimos autorização dos responsáveis.",
    ]
    for segment in result["segments"]:
        assert text[segment["char_start"]:segment["char_end"]] == segment["text"]
        assert segment["n_tokens"] == segmenter.count_tokens(segment["text"])
    notices = result["extraction"]["warnings"]
    assert len(notices) == 3
    excluded = []
    for notice in notices:
        match = re.fullmatch(r"NON_CONTENT_FILTER: intervalo (\d+):(\d+) desconsiderado \(.+\)\.", notice)
        assert match
        excluded.append(text[int(match[1]):int(match[2])])
    assert excluded == ["Alterações nesta política", "Outubro /2026", "ANEXO I 1. POLÍTICA DE PRIVACIDADE DE ALUNOS"]


def test_mixed_paragraph_and_short_chunk_tail_are_not_filtered(segmenter):
    text = "Alterações nesta política\nAvisaremos por e-mail."
    assert prepare(segmenter, text)["segments"][0]["text"] == text
    # O último recorte continua sendo parte do parágrafo, mesmo parecendo um título.
    text = "dados " * 382 + "Alterações nesta política"
    result = prepare(segmenter, text)
    assert len(result["segments"]) == 2
    assert result["segments"][-1]["text"] == "Alterações nesta política"


def test_only_titles_or_dates_returns_clear_error(segmenter):
    with pytest.raises(IngestionError, match="títulos ou datas") as failure:
        prepare(segmenter, "Alterações nesta política\n\nOutubro /2026")
    assert failure.value.code == "UNREADABLE_FILE"


def test_pdf_preserves_pages_and_offsets_after_filtering(segmenter):
    from test_ingestion import pdf_bytes, assert_preserved
    content = pdf_bytes(["Alterações nesta política", "Outubro /2026", "Não vendemos dados."])
    document = extract_document(content, "politica.pdf", document_scope="full_document")
    result = segmenter.prepare(document)
    assert result["extraction"]["status"] == "ok"
    assert result["extraction"]["page_count"] == 3
    assert len(result["extraction"]["warnings"]) == 2
    assert [(s["text"], s["page_start"]) for s in result["segments"]] == [("Não vendemos dados.", 3)]
    assert_preserved(result, segmenter)


def test_analysis_never_classifies_filtered_paragraphs(segmenter, monkeypatch):
    from privacy_service import pipeline
    classified = []

    class Classifier:
        def predict(self, texts, **_kwargs):
            classified.extend(texts)
            return [{"text": text, "n_tokens": segmenter.count_tokens(text),
                     "labels": ["Third Party Sharing/Collection"],
                     "probabilities": {label: float(label == "Third Party Sharing/Collection") for label in LABELS}}
                    for text in texts]

    analysis = AnalysisPipeline.__new__(AnalysisPipeline)
    analysis.settings = SimpleNamespace(limits=Settings.from_env().limits, batch_size=4)
    analysis.segmenter = segmenter
    analysis.classifier = Classifier()
    analysis.rules = load_rules(Settings.from_env().scoring_rules)
    analysis.model_metadata = None
    analysis.recommendations = None
    analysis.versions = {}
    monkeypatch.setattr(pipeline.recommendations, "apply_recommendations", lambda evaluation, _: evaluation)
    statement = "Não compartilhamos seus dados."
    text = f"Alterações nesta política\n\nOutubro /2026\n\n{statement}"
    result = analysis.analyze(text.encode(), "politica.txt", "full_document", time.monotonic() + 30)["result"]
    assert classified == [statement]
    assert len(result["segments"]) == 1
    assert result["evaluation"]["score_blockers"] == []
    baseline = analysis.analyze(statement.encode(), "politica.txt", "full_document", time.monotonic() + 30)["result"]
    assert result["evaluation"]["score"] == baseline["evaluation"]["score"]
    assert [(d["id"], d["score"]) for d in result["evaluation"]["dimensions"]] == [
        (d["id"], d["score"]) for d in baseline["evaluation"]["dimensions"]]
