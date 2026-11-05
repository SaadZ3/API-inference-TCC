"""Registra a versão congelada da R7, preservando os exemplos históricos."""
import json
from pathlib import Path
from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.scoring_common import content_hash
from privacy_service.pipeline import evaluate

ROOT = Path(__file__).resolve().parents[1]
rules = load_rules(ROOT / 'config/scoring.yaml')
assert rules.data['version'] == 'governance-rules-v3-dev2'
target = ROOT / 'tests/fixtures/r7'
target.mkdir(exist_ok=False)
for name in ('complete', 'partial', 'conflict', 'truncated', 'empty', 'unicode'):
    data = json.loads((ROOT / f'tests/fixtures/r6/{name}.json').read_text(encoding='utf-8'))
    result = data['result']
    result['evaluation'] = evaluate(result, result['segments'], rules, None)
    data['versions']['scoring'] = rules.data['version'] + ':' + content_hash(rules.data)
    (target / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('6 exemplos atuais da R7; arquivos da R6 preservados.')
