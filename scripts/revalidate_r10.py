"""Recalcula regras sobre trechos salvos; não executa extração ou BERT novamente."""
from pathlib import Path
from copy import deepcopy
import hashlib, json
from importlib.metadata import version
from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.scoring import code_hashes
from privacy_pipeline.scoring_common import content_hash
from privacy_pipeline.recommendations import load_catalog, apply_recommendations
from privacy_service.pipeline import evaluate, file_hash
import privacy_service.pipeline as pipeline

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/evidencias/R10'
RULES=load_rules(ROOT/'tcc-inference/config/scoring.yaml')
CATALOG=load_catalog(ROOT/'tcc-inference/config/recommendations.yaml')
identity={'rules_version':RULES.data['version'],'rules_sha256':content_hash(RULES.data),'rules_file_sha256':file_hash(ROOT/'tcc-inference/config/scoring.yaml'),'rules':RULES.data,'code_sha256':{**code_hashes(),'pipeline.py':file_hash(pipeline.__file__)},'pyyaml':version('PyYAML')}
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def updated(saved,advice=True):
    output=deepcopy(saved)
    output['versions']['scoring']=RULES.data['version']+':'+content_hash(identity)
    output['versions']['recommendations']=CATALOG.identity if advice else None
    result=output['result']
    result['evaluation']=evaluate(result,result['segments'],RULES,None)
    if advice: result['evaluation']=apply_recommendations(result['evaluation'],CATALOG)
    if 'profile_id' in output: output['profile_id']=RULES.data['profile_id']
    return output
summary=[]
for path in sorted((ROOT/'docs/evidencias/R9/politicas').glob('resultado-*.json')):
    saved=json.loads(path.read_text(encoding='utf-8'))
    after=updated(saved)
    before=saved['result']['evaluation']; current=after['result']['evaluation']
    assert [(q['id'],q['status'],q['value']) for d in before['dimensions'] for q in d['questions']]==[(q['id'],q['status'],q['value']) for d in current['dimensions'][:7] for q in d['questions']]
    assert after['result']['segments']==saved['result']['segments']
    active=sum(d['score'] is not None for d in before['dimensions'])
    formula_only=sum(d['score'] or 0 for d in before['dimensions'])/(active+2)
    summary.append({'policy':saved['result']['source']['filename'],'file':path.name,'segments':len(saved['result']['segments']),'before':before['score'],'formula_only_new_categories_zero':formula_only,'after':current['score'],'old_question_states_unchanged':True,'new_dimensions':[{'id':d['id'],'score':d['score'],'questions':[{'id':q['id'],'status':q['status'],'evidence':[e['quote'] for g in q['supporting_groups'] for e in g['evidence']]} for q in d['questions']]} for d in current['dimensions'][7:]],'blockers':current['score_blockers']})
    save(OUT/'politicas'/path.name,after)
save(OUT/'politicas/resumo.json',summary)
save(OUT/'replay-manifest.json',{'mode':'saved_segments_no_new_extraction_or_model','scoring':identity,'recommendations':CATALOG.manifest,'inputs':{p.name:file_hash(p) for p in (ROOT/'docs/evidencias/R9/politicas').glob('resultado-*.json')}})
for name in ['complete','partial','conflict','truncated','empty','unicode']:
    source=json.loads((ROOT/f'tcc-back/tests/fixtures/r9/{name}.json').read_text(encoding='utf-8'))
    save(ROOT/f'tcc-inference/tests/fixtures/r10/{name}.json',updated(source,False))
    save(ROOT/f'tcc-back/tests/fixtures/r10/{name}.json',updated(source))

# Resultados sintéticos para verificar contrato e interface, com probabilidades ilustrativas.
base=json.loads((ROOT/'tcc-back/tests/fixtures/r9/complete.json').read_text(encoding='utf-8'))
def synthetic(text):
    result=deepcopy(base)
    r=result['result']; source=r['source']
    source.update(filename='exemplo-r10.txt',byte_size=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest())
    r['extraction'].update(text=text,pages=[{'page_number':None,'char_start':0,'char_end':len(text),'status':'ok'}])
    segment=deepcopy(r['segments'][0]); segment.update(text=text,char_start=0,char_end=len(text),n_tokens=200)
    r['segments']=[segment]
    return updated(result)
complete=base['result']['extraction']['text']+'\n\nNeste site, respeitamos o sinal Do Not Track: quando ele está ativo, não usamos o histórico de navegação para publicidade personalizada.\n\nO serviço não é destinado a menores de 13 anos.'
for name,text in {'nine-supported':complete,'negative-dnt':'Nosso site não responde ao sinal Do Not Track.','dnt-conflict':'Nosso site responde ao sinal Do Not Track. Nosso site não responde ao sinal Do Not Track.'}.items():
    save(ROOT/f'tcc-back/tests/fixtures/r10/{name}.json',synthetic(text))
(ROOT/'tcc-back/tests/fixtures/r10/policy.txt').write_text(complete,encoding='utf-8')
for item in summary: print(item['policy'],round(item['before'],3),round(item['after'],3),[(d['id'],d['score']) for d in item['new_dimensions']])
