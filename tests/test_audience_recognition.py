"""Confere que "não é direcionado a menores de 13 anos" delimita o público."""
from pathlib import Path

import pytest

from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.scoring_common import LABELS
from privacy_service.extraction import extract_document
from privacy_service.pipeline import evaluate

RULES = load_rules(Path(__file__).resolve().parents[1] / 'config/scoring.yaml')
LABEL = 'International and Specific Audiences'


def scope_question(segmenter, text):
    prepared = segmenter.prepare(extract_document(text.encode('utf-8'), 'politica.txt', document_scope='full_document'))
    segments = [{**s, 'labels': [LABEL], 'probabilities': {l: float(l == LABEL) for l in LABELS}} for s in prepared['segments']]
    result = evaluate(prepared, segments, RULES, None, {'mode': 'selected_categories', 'category_ids': ['audiences']})
    dimension = next(d for d in result['dimensions'] if d['id'] == 'audiences')
    return next(q for q in dimension['questions'] if q['id'] == 'audiences.scope')


@pytest.mark.parametrize('text', [
    'O TikTok não é direcionado a menores de 13 anos.',
    'Nossos serviços não são direcionados a crianças.',
    'O aplicativo é destinado a adolescentes a partir de 16 anos.',
])
def test_directed_audience_is_scope(segmenter, text):
    assert 'audience_scope' in scope_question(segmenter, text)['found_elements']


def test_audience_without_scope_verb_is_not_scope(segmenter):
    assert 'audience_scope' not in scope_question(segmenter, 'Informações relacionadas a adolescentes.')['found_elements']
