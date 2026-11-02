"""Reextrai PDFs e reavalia exportações, preservando as previsões originais."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from privacy_pipeline import heuristics, scoring, recommendations
from privacy_service import extraction

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pdfs = {sha256(p.read_bytes()).hexdigest(): p for p in args.inputs.glob('*.pdf')}
    rules = heuristics.load_rules(ROOT / 'config/scoring.yaml')
    catalog = recommendations.load_catalog(ROOT / 'config/recommendations.yaml')
    summary = []
    exports = sorted(args.inputs.glob('resultado-*.json'))
    if not exports:
        raise ValueError('Nenhuma exportação encontrada.')
    for path in exports:
        saved = json.loads(path.read_text(encoding='utf-8-sig'))
        result = saved['result']
        pdf = pdfs[result['source']['sha256']]
        document = extraction.extract_path(pdf, document_scope=result['document_scope'])
        # Se o texto mudou, as previsões antigas não descrevem a nova entrada.
        if document.text != result['extraction']['text']:
            raise ValueError(f'Texto diferente em {pdf.name}; execute uma análise completa com o modelo.')
        policy = scoring.from_service_response(saved)
        policy['extraction'] = {**result['extraction'], 'status': document.status,
                                'warnings': document.warnings, 'pages': [asdict(p) for p in document.pages]}
        evaluation = recommendations.apply_recommendations(scoring.evaluate_policy(policy, rules), catalog)
        updated = {**result, 'extraction': policy['extraction'], 'evaluation': evaluation}
        record = {
            'source_analysis_id': saved['analysis_id'],
            'validation_mode': 'fresh_extraction_and_rules_with_saved_predictions',
            'original_versions': saved['versions'],
            'code_sha256': {name: sha256(Path(module.__file__).read_bytes()).hexdigest()
                            for name, module in [('extraction', extraction), ('heuristics', heuristics), ('scoring', scoring)]},
            'result': updated,
        }
        (args.output / path.name).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        old = result['evaluation']
        entry = {
            'analysis_id': saved['analysis_id'], 'pdf': pdf.name, 'sha256': document.sha256,
            'text_identical': True, 'previous_blockers': old['score_blockers'],
            'score_blockers': evaluation['score_blockers'], 'score': evaluation['score'],
            'mean': sum(d['contribution'] for d in evaluation['dimensions']) if evaluation['score'] is not None else None,
            'dimensions': {d['id']: d['score'] for d in evaluation['dimensions']},
            'inconclusive': [q['id'] for d in evaluation['dimensions'] for q in d['questions'] if q['status'] == 'inconclusive'],
            'ambiguity_findings': sum(f['kind'] == 'ambiguity' for f in evaluation['findings']),
        }
        summary.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    (args.output / 'resumo.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
