"""Confere o aviso de mudanças após a preparação, inclusive pela busca complementar."""
import copy
from pathlib import Path

import pytest

from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.scoring_common import LABELS
from privacy_service.extraction import extract_document
from privacy_service.pipeline import evaluate

RULES = load_rules(Path(__file__).resolve().parents[1] / 'config/scoring.yaml')
FIRST = 'Atualizaremos esta Política.'
NOTICE = 'Enviaremos aviso por e-mail sobre as alterações antes da entrada em vigor.'


@pytest.mark.parametrize('text,score', [
    (f'{FIRST} {NOTICE}', 3),
    (f'{FIRST}\n{NOTICE}', 3),
    (f'{FIRST} Enviaremos aviso por e-mail sobre as alterações.', 2.25),
    (f'{FIRST} Enviaremos recibos por e-mail.', 0),
    (f'{FIRST} Enviaremos aviso por e-mail sobre as alterações cadastrais.', 0),
    (f'{FIRST}\n\n{NOTICE}', 0),
])
def test_prepared_notice_and_original_evidence(segmenter, text, score):
    prepared = segmenter.prepare(extract_document(
        text.encode('utf-8'), 'politica.txt', document_scope='full_document'))
    segments = [{**segment, 'labels': ['Other'],
                 'probabilities': {label: float(label == 'Other') for label in LABELS}}
                for segment in prepared['segments']]
    before = copy.deepcopy((prepared, segments))
    result = evaluate(prepared, segments, RULES, None,
                      {'mode': 'selected_categories', 'category_ids': ['changes']})
    assert (prepared, segments) == before
    assert result['rules_version'] == 'governance-rules-v6-dev1'
    assert result['score'] == score
    assert result['score_blockers'] == []
    by_id = {s['segment_id']: s for s in segments}
    for dimension in result['dimensions']:
        for question in dimension['questions']:
            for group in question['supporting_groups']:
                assert group['detection_source'] == 'rule_only'
                for evidence in group['evidence']:
                    segment = by_id[evidence['segment_id']]
                    assert segment['text'][evidence['start']:evidence['end']] == evidence['quote']
                    assert prepared['extraction']['text'][segment['char_start']:segment['char_end']] == segment['text']
