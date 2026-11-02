# Verifica carregamento e execução do fluxo de análise para proteger o comportamento do TCC.
import time

import pytest

from privacy_pipeline.scoring_common import LABELS
from privacy_pipeline.heuristics import load_rules
from privacy_service.pipeline import evaluate, check_deadline
from privacy_service.settings import Settings


@pytest.mark.parametrize("scope,status,expected,blockers", [
    ("full_document", "ok", "complete_input", []),
    ("excerpt", "ok", "partial_input", ["document_scope_incomplete"]),
    ("unknown", "ok", "partial_input", ["document_scope_incomplete"]),
    ("full_document", "partial", "partial_input", ["extraction_partial"]),
    ("excerpt", "partial", "partial_input", ["document_scope_incomplete", "extraction_partial"]),
    ("unknown", "partial", "partial_input", ["document_scope_incomplete", "extraction_partial"]),
])
def test_score_completeness(model_dir, scope, status, expected, blockers):
    rules = load_rules(Settings.from_env().scoring_rules)
    prepared = {"source": {"sha256": "policy"}, "document_scope": scope, "extraction": {"status": status}}
    segment = {"segment_id": "one", "text": "Política de privacidade.", "labels": [],
               "probabilities": {label: 0.0 for label in LABELS}, "n_tokens": 8, "truncated": False}
    result = evaluate(prepared, [segment], rules, None)
    assert result["input_status"] == expected
    assert result["score_blockers"] == blockers
    assert (result["score"] is not None) == (expected == "complete_input")
    segment["truncated"] = True
    result = evaluate(prepared, [segment], rules, None)
    assert result["input_status"] == "no_usable_segments"
    assert result["score"] is None


def test_deadline():
    with pytest.raises(TimeoutError):
        check_deadline(time.monotonic() - 1)
    check_deadline(time.monotonic() + 1)


def test_bad_export_hash_is_rejected_before_loading(tmp_path):
    import json
    from privacy_service.pipeline import AnalysisPipeline, ARTIFACTS
    for name in ARTIFACTS:
        (tmp_path / name).write_bytes(b"invalid")
    (tmp_path / "checksums.json").write_text(json.dumps({name: "0" * 64 for name in ARTIFACTS}))
    with pytest.raises(ValueError, match="Integridade"):
        AnalysisPipeline(Settings(tmp_path, tmp_path / "missing_rules.yaml"))
