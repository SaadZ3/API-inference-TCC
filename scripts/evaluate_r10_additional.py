from pathlib import Path
import json,hashlib,sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/tests'))
from test_governance_scoring import policy,question,verify_evidence
from privacy_pipeline.heuristics import load_rules
from privacy_pipeline.scoring import evaluate_policy
out=root/'docs/evidencias/R10'
frozen=json.loads((out/'freeze.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==sha for p,sha in frozen['files_sha256'].items())
source=out/'additional-cases.json'
cases=json.loads(source.read_text(encoding='utf-8'))['cases']
rules=load_rules(root/'tcc-inference/config/scoring.yaml')
results=[]
for case in cases:
    inp=policy(case['text']); result=evaluate_policy(inp,rules);verify_evidence(result,inp)
    actual=question(result,case['question'])['status']
    results.append({**case,'actual':actual,'matches_expectation':actual==case['expected']})
output={'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'freeze':frozen,'passed':sum(r['matches_expectation'] for r in results),'total':len(results),'results':results}
(out/'additional-results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(output['passed'], '/',output['total'])
for case in results:
    if not case['matches_expectation']:print(case['id'],case['expected'],case['actual'])
