"""Deterministic advice attached to findings, independent of the documentary score."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import yaml
from .scoring_common import content_hash
from .heuristics import PROFILES

RANK = {'high': 0, 'medium': 1, 'low': 2}


@dataclass(frozen=True)
class Catalog:
    data: dict
    references: dict
    manifest: dict
    identity: str


def load_catalog(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if data.get('profile_id') not in PROFILES or not data.get('version'):
        raise ValueError('Catálogo incompatível com o perfil.')
    source_path = path.parent / data['sources_file']
    if source_path.resolve().parent != path.resolve().parent:
        raise ValueError('Referências devem estar na pasta do catálogo.')
    sources = json.loads(source_path.read_text(encoding='utf-8'))
    refs = {r['id']: r for r in sources['references']}
    if len(refs) != len(sources['references']) or 'project-review' not in refs:
        raise ValueError('Referências duplicadas ou rubrica ausente.')
    for ref in refs.values():
        if (ref['source_type'] not in {'public_guidance', 'project_rubric'}
                or not all(ref.get(k) for k in ['verified_by', 'verified_at', 'source_locator', 'summary'])
                or len(ref['source_sha256']) != 64 or ref['pdf_page'] is not None
                or (ref['source_type'] == 'public_guidance' and
                    (ref['verification_scope'] != 'provided_workbook' or not ref['url'].startswith('https://www.isms.online/')))):
            raise ValueError('Referência incompleta ou origem não conferida.')
    for topic in data['topics'].values():
        for references in topic['references'].values():
            if not references or any(r not in refs for r in references):
                raise ValueError('Controle sem referência conferida.')
        for target in ('organizational_verification', 'policy_wording'):
            if any(not topic[target].get(key) for key in ('what_to_do', 'how_to_do', 'completion_evidence')):
                raise ValueError('Orientação incompleta.')
    expected_questions = {q for topic in data['topics'].values() for q in topic['references']}
    if set(data.get('questions', {})) != expected_questions or expected_questions != PROFILES[data['profile_id']][1]:
        raise ValueError('Catálogo deve cobrir cada pergunta do perfil.')
    for template in data['questions'].values():
        if any(not template.get(key) for key in ('what_to_do', 'how_to_do', 'completion_evidence')):
            raise ValueError('Orientação da pergunta incompleta.')
    manifest = {'version': data['version'], 'catalog_sha256': content_hash(data),
                'catalog_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'sources_sha256': content_hash(sources),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'catalog': data, 'sources': sources}
    return Catalog(data, refs, manifest, data['version'] + ':' + content_hash(manifest))


def apply_recommendations(evaluation, catalog):
    """Copy the result; only recommendation links/list and their limitation change."""
    if evaluation['profile_id'] != catalog.data['profile_id']:
        raise ValueError('Perfil não suportado pelo catálogo.')
    result = deepcopy(evaluation)
    result['recommendations'] = []
    for finding in result['findings']:
        finding['recommendation_ids'] = []
    questions = {q['id']: q for d in result['dimensions'] for q in d['questions']}

    def append(template_id, target, findings, template, refs, role, pending):
        if not findings:
            return
        findings = sorted(findings, key=lambda f: f['id'])
        finding_ids = [f['id'] for f in findings]
        priority = min((f['priority'] for f in findings), key=RANK.get)
        refs = [deepcopy(catalog.references[key]) for key in sorted(set(refs))]
        controller = any(r['control_role'] == 'controller' for r in refs)
        recommendation = {
            'id': 'rec-' + content_hash({'catalog': catalog.identity, 'template': template_id, 'findings': finding_ids})[:24],
            'template_id': template_id, 'catalog_version': catalog.identity,
            'finding_ids': finding_ids, 'target': target,
            'trigger_reason': 'Revisão sugerida pelos achados: ' + '; '.join(
                (questions[f['question_id']]['title'] + ': ' if f['question_id'] in questions else '') + f['rationale']
                for f in findings),
            'what_to_do': template['what_to_do'], 'how_to_do': list(template['how_to_do']),
            'completion_evidence': list(template['completion_evidence']), 'suggested_role': role,
            'references': refs, 'pending_fields': list(pending), 'priority': priority,
            'priority_reason': 'Ordem sugerida herdada dos achados vinculados: ' + '; '.join(sorted({f['priority_reason'] for f in findings})),
            'applicability_note': (
                'Referências para controlador: confirme o papel e o contexto da organização antes de aplicar. '
                if controller else 'Confirme o contexto e a pertinência antes de aplicar. ') +
                'Associação temática para revisão; não é equivalência entre OPP-115, ISO e LGPD.',
        }
        result['recommendations'].append(recommendation)
        for finding in findings:
            finding['recommendation_ids'].append(recommendation['id'])

    for dimension, topic in catalog.data['topics'].items():
        for question_id, references in topic['references'].items():
            if question_id not in questions:
                continue
            for kind, target in [('information_gap', 'policy_wording'), ('practice_statement', 'organizational_verification')]:
                findings = [f for f in result['findings'] if f['dimension_id'] == dimension
                            and f['question_id'] == question_id and f['kind'] == kind]
                # Uma informação encontrada em outro trecho não vira lacuna local.
                if kind == 'information_gap' and questions[question_id]['status'] not in {'partial', 'not_detected'}:
                    continue
                template = catalog.data['questions'][question_id] if kind == 'information_gap' else topic[target]
                append(question_id+'.'+target, target, findings, template, references,
                       topic['suggested_role'], topic['pending_fields'])
    for key, template in catalog.data['special'].items():
        findings = [f for f in result['findings'] if f['kind'] in template['kinds']]
        groups = sorted({f['question_id'] or '' for f in findings})
        for question_id in groups:
            linked = [f for f in findings if (f['question_id'] or '') == question_id]
            template_id = 'review.'+key+('.'+question_id if question_id else '')
            append(template_id, 'organizational_verification', linked, template, ['project-review'],
                   'Revisor e responsável pelo documento', template['pending_fields'])
    result['recommendations'].sort(key=lambda r: (RANK[r['priority']], r['template_id']))
    result['limitations'] = [s for s in result['limitations'] if not s.startswith(('Catálogo de recomendações ainda não aplicado;', 'Recomendações de desenvolvimento:'))]
    result['limitations'].append('Recomendações de desenvolvimento: guias da planilha conferidos, sem conferência do texto integral da ISO e sem revisão técnica humana independente. Não comprovam atendimento aos controles.')
    return result


def render_recommendations(result):
    lines = ['\n## Recomendações para revisão', '']
    for item in result['recommendations']:
        lines += [f'### {item["what_to_do"]}', '', item['trigger_reason'], '',
                  f'Alvo: {item["target"]}. Prioridade sugerida: {item["priority"]}.',
                  item['applicability_note'], '', '**Como fazer**',
                  *[f'{i+1}. {step}' for i, step in enumerate(item['how_to_do'])], '',
                  '**Como verificar**', *['- '+step for step in item['completion_evidence']], '',
                  'Informações a confirmar: ' + '; '.join(item['pending_fields']), '',
                  *[f'- {r["title"]}; {r["edition"]}; {r["section"]}; {r["source_locator"]}. Fonte: {r["source_file"]}. ' +
                    (r['url'] or '') for r in item['references']]]
    if not result['recommendations']:
        lines.append('Catálogo aplicado; nenhum template acionado. Isso não comprova conformidade.')
    return '\n'.join(lines)+'\n'
