"""Exporta casos do cálculo parcial sem alterar os exemplos históricos."""
import copy
import json
from importlib.metadata import version
from pathlib import Path

from privacy_pipeline import scoring, scoring_common
from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.recommendations import apply_recommendations, load_catalog
from privacy_service import pipeline
from privacy_service.pipeline import evaluate, file_hash

ROOT = Path(__file__).resolve().parents[2]
RULES = load_rules(ROOT / 'tcc-inference/config/scoring.yaml')
CATALOG = load_catalog(ROOT / 'tcc-inference/config/recommendations.yaml')
DIRECTORY = ROOT / 'tcc-back/tests/fixtures'
IDENTITY = {
    'rules_version': RULES.data['version'],
    'rules_sha256': scoring_common.content_hash(RULES.data),
    'rules_file_sha256': file_hash(ROOT / 'tcc-inference/config/scoring.yaml'),
    'rules': RULES.data,
    'code_sha256': {**scoring.code_hashes(), 'pipeline.py': file_hash(pipeline.__file__)},
    'pyyaml': version('PyYAML'),
}

for name, original in [('excerpt', 'two'), ('unknown', 'two'), ('partial', 'two'),
                       ('all-inconclusive', 'all-inconclusive'), ('truncated', 'truncated'), ('empty', 'empty')]:
    value = json.loads((DIRECTORY / f'r11/{original}.json').read_text(encoding='utf-8'))
    result = value['result']
    selection = copy.deepcopy(result['evaluation']['assessment_scope'])
    if selection['mode'] == 'all_categories':
        selection = {'mode': 'selected_categories', 'category_ids': ['retention', 'security']}
    result['document_scope'] = 'unknown' if name == 'unknown' else 'excerpt'
    if name == 'partial':
        result['extraction']['status'] = 'partial'
    result['evaluation'] = apply_recommendations(evaluate(result, result['segments'], RULES, None, selection), CATALOG)
    value['versions']['scoring'] = RULES.data['version'] + ':' + scoring_common.content_hash(IDENTITY)
    value['versions']['recommendations'] = CATALOG.identity
    target = DIRECTORY / f'selected-average/{name}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

print('Exemplos da média parcial exportados; resultados históricos preservados.')
