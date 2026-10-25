"""Exemplos sintéticos com regras reais, sem executar o classificador."""
import json
from pathlib import Path

from privacy_pipeline.recommendations import apply_recommendations, load_catalog
from export_e4_fixtures import response, rules
from privacy_service.pipeline import evaluate

ROOT = Path(__file__).resolve().parents[2]
catalog = load_catalog(ROOT / 'OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/config/recommendations.yaml')
target = ROOT / 'tcc-back/tests/fixtures/segment-guidance'
target.mkdir(exist_ok=True)
samples = {name: json.loads((ROOT / f'tcc-back/tests/fixtures/e4/{name}.json').read_text(encoding='utf-8'))
           for name in ['complete', 'conflict', 'unicode']}
samples['local'] = response('🔒 Guardamos dados pelo tempo necessário.\n\nCompartilhamos informações com terceiros.')
for segment, label in zip(samples['local']['result']['segments'], ['Data Retention', 'Third Party Sharing/Collection'], strict=True):
    segment['labels'] = [label]
    segment['probabilities'] = {key: .99 if key == label else .01 for key in segment['probabilities']}
parts = ['Para solicitar exclusão, escreva para', 'privacidade@exemplo.com.']
samples['complement'] = response(' '.join(parts))
base = samples['complement']['result']['segments'][0]
offset = 0
samples['complement']['result']['segments'] = []
for i, text in enumerate(parts):
    samples['complement']['result']['segments'].append({**base, 'segment_id': f's{i}', 'text': text,
                                                       'char_start': offset, 'char_end': offset + len(text)})
    offset += len(text) + 1
for name in ['local', 'complement']:
    prepared = samples[name]['result']
    prepared['evaluation'] = evaluate(prepared, prepared['segments'], rules, None)
for name, data in samples.items():
    data['result']['evaluation'] = apply_recommendations(data['result']['evaluation'], catalog)
    data['versions']['recommendations'] = catalog.identity
    (target / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'{len(samples)} exemplos exportados com rótulos sintéticos e recomendações por pergunta.')
