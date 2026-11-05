"""Gera casos da R6 sem reescrever o histórico e sem executar o classificador."""
import json
from pathlib import Path

from export_e4_fixtures import response, rules
from privacy_pipeline.recommendations import apply_recommendations, load_catalog
from privacy_pipeline.scoring_common import content_hash
from privacy_service.pipeline import evaluate

ROOT = Path(__file__).resolve().parents[2]
catalog = load_catalog(ROOT / 'tcc-inference/config/recommendations.yaml')
if rules.data['version'] != 'governance-rules-v3-dev1':
    raise RuntimeError('A R6 está congelada. Use outro diretório para a versão atual.')
service = ROOT / 'tcc-inference/tests/fixtures/r6'
backend = ROOT / 'tcc-back/tests/fixtures/r6'
service.mkdir(exist_ok=True)
backend.mkdir(exist_ok=True)
samples = {}
for name in ('complete', 'partial', 'conflict', 'truncated', 'empty', 'unicode'):
    data = json.loads((ROOT / f'tcc-inference/tests/fixtures/r3/{name}.json').read_text(encoding='utf-8'))
    result = data['result']
    result['evaluation'] = evaluate(result, result['segments'], rules, None)
    data['versions']['scoring'] = rules.data['version'] + ':' + content_hash(rules.data)
    data['versions']['recommendations'] = None
    (service / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    samples[name] = data
samples['partial-credit'] = response('Você pode solicitar exclusão.')
samples['use-purpose'] = response('Usamos seu e-mail para enviar recibos.')
samples['local'] = response('🔒 Guardamos dados pelo tempo necessário.\n\nCompartilhamos informações com terceiros.')
for data in samples.values():
    data['versions']['recommendations'] = catalog.identity
    data['result']['evaluation'] = apply_recommendations(data['result']['evaluation'], catalog)
for name, data in samples.items():
    (backend / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'{len(samples)} casos da R6 com catálogo e referências; histórico preservado.')
