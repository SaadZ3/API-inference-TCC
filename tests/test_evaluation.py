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
    saved = json.loads((ROOT / f'tests/fixtures/r1/{name}.json').read_text(encoding='utf-8'))['result']
    before = copy.deepcopy(saved)
    actual = evaluate(saved, saved['segments'], RULES, None)
    assert actual == saved['evaluation']
    assert saved == before
    # A correção não muda as notas nem os estados dos exemplos históricos.
    historical = json.loads((ROOT / f'tests/fixtures/e4/{name}.json').read_text(encoding='utf-8'))['result']['evaluation']
    assert actual['score'] == historical['score']
    assert actual['score_blockers'] == historical['score_blockers']
    assert [(q['id'], q['status']) for d in actual['dimensions'] for q in d['questions']] == [
        (q['id'], q['status']) for d in historical['dimensions'] for q in d['questions']]
