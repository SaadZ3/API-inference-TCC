"""Fixtures de contrato com motor real e categorias sintéticas explicitadas."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from privacy_pipeline.scoring_common import LABELS, content_hash
from privacy_pipeline.heuristics import load_rules
from privacy_service.pipeline import evaluate

ROOT = Path(__file__).resolve().parents[2]
rules = load_rules(ROOT / 'OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/config/scoring.yaml')


def response(text, scope='full_document', truncated=False):
    raw = text.encode('utf-8')
    parts, offset = [], 0
    for i, paragraph in enumerate(text.split('\n\n')):
        if paragraph:
            parts.append({'segment_id': f's{i}', 'text': paragraph, 'char_start': offset, 'char_end': offset+len(paragraph),
                          'page_start': None, 'page_end': None, 'labels': ['Other'], 'n_tokens': 20,
                          'probabilities': {label: .99 if label == 'Other' else .01 for label in LABELS}, 'truncated': truncated})
        offset += len(paragraph)+2
    prepared = {'document_scope': scope,
                'source': {'filename': 'synthetic-e4.txt', 'format': 'txt', 'byte_size': max(1,len(raw)), 'sha256': sha256(raw).hexdigest()},
                'preprocessing': {'version': 'synthetic-fixture-not-inference', 'max_length': 512, 'tokenizer_sha256': '0'*64},
                'extraction': {'status': 'ok', 'warnings': [], 'text': text, 'page_count': None,
                               'pages': [{'page_number': None, 'char_start': 0, 'char_end': len(text), 'status': 'ok'}]}}
    result = {**prepared, 'segments': parts, 'evaluation': evaluate(prepared, parts, rules, None)}
    return {'schema_version': 2, 'versions': {'model': 'synthetic-labels-not-model-inference',
            'preprocessing': 'synthetic-fixture', 'scoring': rules.data['version']+':'+content_hash(rules.data), 'recommendations': None}, 'result': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)
    manual = json.loads((ROOT / 'OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/tests/fixtures/governance/manual_policies.json').read_text(encoding='utf-8'))
    text = manual['policies'][0]['text']
    cases = {'complete': response(text), 'partial': response(text, 'excerpt'),
             'conflict': response('Coletamos nome. Não coletamos nome.'),
             'truncated': response(text, truncated=True), 'empty': response(''),
             'unicode': response('🔒 Para solicitar exclusão, escreva para privacidade@exemplo.com.')}
    for name, value in cases.items():
        (out / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'{len(cases)} synthetic fixtures exported; evaluator, no model inference.')


if __name__ == '__main__':
    main()
