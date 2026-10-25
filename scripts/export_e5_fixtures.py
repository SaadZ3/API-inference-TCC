"""E5 contract samples derived from E4 fixtures; never rewrite historical samples."""
import json
from pathlib import Path
from privacy_pipeline.recommendations import load_catalog, apply_recommendations
from export_e4_fixtures import response

ROOT = Path(__file__).resolve().parents[2]
catalog = load_catalog(ROOT / 'OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/config/recommendations.yaml')
target = ROOT / 'tcc-back/tests/fixtures/e5'
target.mkdir(exist_ok=True)
samples = {source.stem: json.loads(source.read_text(encoding='utf-8'))
           for source in (ROOT / 'tcc-back/tests/fixtures/e4').glob('*.json')}
samples['practice'] = response('Retemos dados pessoais indefinidamente. Não criptografamos os dados transmitidos.')
for name, data in samples.items():
    data['result']['evaluation'] = apply_recommendations(data['result']['evaluation'], catalog)
    data['versions']['recommendations'] = catalog.identity
    (target / (name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Exported {len(samples)} E5 samples with synthetic predictions, catalog '+catalog.identity)
