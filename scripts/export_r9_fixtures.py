"""Produz exemplos da R9 e reavalia os mesmos segmentos reais, preservando R7/R8."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
from export_e4_fixtures import response, rules
from privacy_pipeline.recommendations import apply_recommendations, load_catalog
from privacy_service.pipeline import evaluate
from privacy_pipeline.scoring import from_service_response, evaluate_policy
from privacy_pipeline.scoring_common import content_hash

ROOT=Path(__file__).resolve().parents[2]
refresh = '--refresh' in sys.argv
catalog=load_catalog(ROOT/'tcc-inference/config/recommendations.yaml')
out=ROOT/'tcc-inference/tests/fixtures/r9'
out.mkdir(exist_ok=refresh)
samples={}
for name in ('complete','partial','conflict','truncated','empty','unicode'):
    data=json.loads((ROOT/f'tcc-inference/tests/fixtures/r7/{name}.json').read_text(encoding='utf-8'))
    data['result']['evaluation']=evaluate(data['result'],data['result']['segments'],rules,None)
    data['versions']['scoring']=rules.data['version']+':'+content_hash(rules.data)
    (out/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    samples[name]=data
samples['independent']=response('Coletamos e-mail para enviar recibos.\n\nColetamos nome. Não coletamos nome.')
def ambiguous(window,q,rules,cache):
    return dict(window=window,state='inconclusive',polarity='uncertain',found=[],missing=q['elements'],features={})
# Isola a agregação quando todos os detectores retornam dúvida; não testa reconhecimento linguístico.
with patch('privacy_pipeline.heuristics.inspect',ambiguous):
    samples['all-inconclusive']=response('Informações dependem de esclarecimento.')
backend=ROOT/'tcc-back/tests/fixtures/r9'
backend.mkdir(exist_ok=refresh)
for name,data in samples.items():
    data['versions']['recommendations']=catalog.identity
    data['result']['evaluation']=apply_recommendations(data['result']['evaluation'],catalog)
    (backend/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
real=ROOT/'docs/evidencias/R9/politicas'
real.mkdir(exist_ok=refresh)
summary=[]
for path in (ROOT/'docs/evidencias/R7/politicas').glob('resultado-*.json'):
    data=json.loads(path.read_text(encoding='utf-8'))
    before=data['result']['evaluation']
    updated=apply_recommendations(evaluate_policy(from_service_response(data),rules),catalog)
    data['result']['evaluation']=updated
    data['versions']['scoring']='r9-replay:'+content_hash({'rules':rules.data,'core':(ROOT/'tcc-inference/vendor/SNAPSHOT.json').read_text(encoding='utf-8')})
    data['versions']['recommendations']=catalog.identity
    (real/path.name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary.append(dict(policy=data['result']['source']['filename'],segments=len(data['result']['segments']),
        before=before['score'],after=updated['score'],blockers=updated['score_blockers'],
        dimensions=[{k:d[k] for k in ('id','score','resolved_questions','excluded_questions','normalized_weight')} for d in updated['dimensions']],
        excluded_evidence=len({(e['segment_id'],e['start'],e['end']) for d in updated['dimensions'] for q in d['questions'] for g in q['contextual_groups'] for e in g['evidence']})))
(real/'resumo.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True))
