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

ROOT = Path(__file__).resolve().parents[2]
RULES = load_rules(ROOT / 'OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/config/scoring.yaml')


@pytest.mark.parametrize('name', ['complete', 'partial', 'conflict', 'truncated', 'empty', 'unicode'])
def test_service_matches_cli_calculation(name):
    saved = json.loads((ROOT / f'tcc-back/tests/fixtures/e4/{name}.json').read_text(encoding='utf-8'))['result']
    before = copy.deepcopy(saved)
    actual = evaluate(saved, saved['segments'], RULES, None)
    assert actual == saved['evaluation']
    assert saved == before
