# Verifica integração das regras e recomendações na API para proteger o comportamento do TCC.
import copy
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest
from privacy_pipeline.heuristics import load_rules
from privacy_service.pipeline import evaluate
from privacy_service.main import create_app
from privacy_service.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules(ROOT / 'config/scoring.yaml')


@pytest.mark.parametrize('name', ['complete', 'partial', 'conflict', 'truncated', 'empty', 'unicode'])
def test_service_matches_cli_calculation(name):
    saved = json.loads((ROOT / f'tests/fixtures/r10/{name}.json').read_text(encoding='utf-8'))['result']
    before = copy.deepcopy(saved)
    actual = evaluate(saved, saved['segments'], RULES, None)
    assert actual == saved['evaluation']
    assert saved == before
    assert actual['uncertainty_policy'] == 'exclude_inconclusive_evidence_v1'
    if name == 'conflict':
        assert actual['score'] == 0
        assert actual['score_blockers'] == []
        assert next(d for d in actual['dimensions'] if d['id']=='collection')['normalized_weight'] == 0
        return
    # As notas novas são contínuas; os estados desses exemplos permanecem iguais.
    historical = json.loads((ROOT / f'tests/fixtures/r9/{name}.json').read_text(encoding='utf-8'))['result']['evaluation']
    assert actual['score_scale'] == 'continuous_0_3'
    expected = historical['score'] * 7 / 9 if historical['score'] is not None else None
    assert actual['score'] == pytest.approx(expected)
    assert actual['score_blockers'] == historical['score_blockers']
    assert len(actual['dimensions']) == 9 and actual['profile_id'] == 'documentary-18-v1'
    assert [(q['id'], q['status']) for d in actual['dimensions'][:7] for q in d['questions']] == [
        (q['id'], q['status']) for d in historical['dimensions'] for q in d['questions']]
