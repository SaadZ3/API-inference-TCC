"""Confere que um endereço sem https:// também conta como canal localizável."""
from pathlib import Path

import pytest

from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.scoring_common import LABELS
from privacy_service.extraction import extract_document
from privacy_service.pipeline import evaluate

RULES = load_rules(Path(__file__).resolve().parents[1] / 'config/scoring.yaml')
REQUEST = 'Para exercer seus direitos de privacidade, acesse a página de Dados e privacidade em'


def channel_status(segmenter, text):
    prepared = segmenter.prepare(extract_document(
        text.encode('utf-8'), 'politica.txt', document_scope='full_document'))
    segments = [{**segment, 'labels': ['User Access, Edit and Deletion'],
                 'probabilities': {label: float(label == 'User Access, Edit and Deletion') for label in LABELS}}
                for segment in prepared['segments']]
    result = evaluate(prepared, segments, RULES, None,
                      {'mode': 'selected_categories', 'category_ids': ['access']})
    access = next(d for d in result['dimensions'] if d['id'] == 'access')
    return next(q for q in access['questions'] if q['id'] == 'access.channel')


@pytest.mark.parametrize('address', [
    'privacy.apple.com/br', 'www.exemplo.com.br', 'https://exemplo.com/privacidade', 'privacidade@exemplo.com',
])
def test_locatable_channel_with_or_without_protocol(segmenter, address):
    question = channel_status(segmenter, f'{REQUEST} {address}.')
    assert 'locatable_channel' in question['found_elements']


@pytest.mark.parametrize('text', [
    f'{REQUEST} nosso site.',
    f'{REQUEST} nosso site, conforme a Lei nº 12.965/14.',
])
def test_text_without_address_is_not_a_channel(segmenter, text):
    assert 'locatable_channel' not in channel_status(segmenter, text)['found_elements']
