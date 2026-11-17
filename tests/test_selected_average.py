"""Confere a média parcial, os bloqueios de leitura e a regra histórica."""
import copy
import json
from pathlib import Path

import pytest

from privacy_pipeline.heuristics import RuleSet, load_rules
from privacy_service.pipeline import evaluate

ROOT = Path(__file__).resolve().parents[1]
RULES = load_rules(ROOT / 'config/scoring.yaml')
SELECTION = {'mode': 'selected_categories', 'category_ids': ['retention', 'security']}


def source(name='two'):
    return json.loads((ROOT.parent / f'tcc-back/tests/fixtures/r11/{name}.json').read_text(encoding='utf-8'))['result']


@pytest.mark.parametrize('scope', ['excerpt', 'unknown'])
def test_incomplete_document_averages_only_selected_categories(scope):
    prepared = source()
    prepared['document_scope'] = scope
    before = copy.deepcopy(prepared)
    result = evaluate(prepared, prepared['segments'], RULES, None, SELECTION)
    assert prepared == before
    assert result['score'] == pytest.approx(1.875)
    assert result['score_status'] == 'available' and result['score_blockers'] == []
    assert result['input_status'] == 'partial_input'
    assert not any(f['kind'] == 'extraction_limitation' for f in result['findings'])
    assert all(d['contribution'] is None and d['normalized_weight'] == 0
               for d in result['dimensions'] if d['id'] not in SELECTION['category_ids'])
    assert any('não representa a política inteira' in text for text in result['limitations'])


@pytest.mark.parametrize('kind,blocker', [
    ('partial', 'extraction_partial'), ('truncated', 'truncated_segments'),
    ('empty', 'no_usable_segments'), ('all-inconclusive', 'no_evaluable_questions'),
])
def test_selected_average_keeps_other_blockers(kind, blocker):
    prepared = source('two' if kind == 'partial' else kind)
    prepared['document_scope'] = 'excerpt'
    if kind == 'partial':
        prepared['extraction']['status'] = 'partial'
    selection = prepared['evaluation']['assessment_scope']
    if selection['mode'] == 'all_categories':
        selection = SELECTION
    result = evaluate(prepared, prepared['segments'], RULES, None, selection)
    assert result['score'] is None and result['score_status'] == 'blocked'
    assert blocker in result['score_blockers']
    assert 'document_scope_incomplete' not in result['score_blockers']


def test_old_method_and_all_categories_still_require_complete_document():
    prepared = source()
    prepared['document_scope'] = 'excerpt'
    historical = copy.deepcopy(RULES.data)
    historical['method_version'] = 'documentary-quality-v5-draft1'
    for rules, selection in [(RuleSet(historical), SELECTION), (RULES, None)]:
        result = evaluate(prepared, prepared['segments'], rules, None, selection)
        assert result['score'] is None
        assert result['score_blockers'] == ['document_scope_incomplete']
