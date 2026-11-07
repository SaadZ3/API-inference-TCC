"""Gera exemplos controlados da seleção R11; não executa o classificador BERT."""
from pathlib import Path
import json,hashlib,copy
from privacy_pipeline.assessment_scope import resolve_assessment_scope
from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.recommendations import load_catalog,apply_recommendations
from privacy_pipeline.scoring import code_hashes
from privacy_pipeline.scoring_common import content_hash
from privacy_service.pipeline import evaluate,file_hash
import privacy_service.pipeline as pipeline
ROOT=Path(__file__).resolve().parents[2]
rules=load_rules(ROOT/'tcc-inference/config/scoring.yaml');catalog=load_catalog(ROOT/'tcc-inference/config/recommendations.yaml')
identity={'rules':rules.data,'code_sha256':{**code_hashes(),'pipeline.py':file_hash(pipeline.__file__)},'selection_contract':3}
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def updated(value,selection):
    out=copy.deepcopy(value);out['schema_version']=3;r=out['result']
    out['versions']['scoring']='governance-rules-v5-dev1:'+content_hash(identity);out['versions']['recommendations']=catalog.identity
    r['evaluation']=apply_recommendations(evaluate(r,r['segments'],rules,None,selection),catalog)
    return out
for name in ['complete','partial','conflict','truncated','empty','unicode']:
    old=json.loads((ROOT/f'tcc-back/tests/fixtures/r10/{name}.json').read_text(encoding='utf-8'))
    save(ROOT/f'tcc-back/tests/fixtures/r11/{name}.json',updated(old,resolve_assessment_scope()))
base=json.loads((ROOT/'tcc-back/tests/fixtures/r10/complete.json').read_text(encoding='utf-8'))
TEXT='Usamos TLS para proteger os dados pessoais durante a transmissão.\n\nFuncionários acessam os registros de clientes somente após autenticação de múltiplos fatores.\n\nGuardamos dados pessoais.'
def synthetic(text,ids,scope='full_document'):
    value=copy.deepcopy(base);r=value['result'];r['document_scope']=scope
    r['source'].update(filename='exemplo-r11.txt',byte_size=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest())
    r['extraction'].update(text=text,pages=[{'page_number':None,'char_start':0,'char_end':len(text),'status':'ok'}])
    segment=copy.deepcopy(r['segments'][0]);segment.update(text=text,char_start=0,char_end=len(text),n_tokens=90)
    r['segments']=[segment]
    return updated(value,resolve_assessment_scope() if ids is None else resolve_assessment_scope({'mode':'selected_categories','category_ids':ids}))
cases={'all':(TEXT,None),'one':(TEXT,['security']),'two':(TEXT,['security','retention']),'three':(TEXT,['security','retention','access']),
       'excluded':('Coletamos nome. Não coletamos nome.\n\n'+TEXT,['collection','security']),
       'all-inconclusive':('Coletamos nome. Não coletamos nome.\n\n'+TEXT,['collection'])}
for name,args in cases.items():save(ROOT/f'tcc-back/tests/fixtures/r11/{name}.json',synthetic(*args))
for scope in ['excerpt','unknown']:save(ROOT/f'tcc-back/tests/fixtures/r11/{scope}.json',synthetic(TEXT,['security','retention'],scope))
(ROOT/'tcc-back/tests/fixtures/r11/policy.txt').write_text(TEXT,encoding='utf-8')
save(ROOT/'docs/evidencias/R11/replay-manifest.json',identity)
print('Exemplos v3 com seleção e notas distintas gerados.')
