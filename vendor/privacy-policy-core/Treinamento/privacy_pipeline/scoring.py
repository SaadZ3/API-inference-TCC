"""Avaliação da qualidade da informação documentada, sem dependência de PyTorch."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re

from . import heuristics
from .scoring_common import SCORE_SCALE, continuous_score, content_hash, validate_policy

VALUES = {"supported": 1.0, "partial": 0.5, "not_detected": 0.0, "inconclusive": None}


def from_service_response(response: dict) -> dict:
    """Adapta uma resposta salva da P4, sem alterar suas previsões ou offsets."""
    result = response["result"]
    return {"schema_version": 1, "policy_id": result["source"]["sha256"], "language": "pt",
            "complete": result["document_scope"] == "full_document",
            "document_scope": result["document_scope"], "extraction": result["extraction"],
            "model": result["evaluation"].get("model"), "applicability": {}, "segments": result["segments"]}


def validate_input(policy: dict) -> None:
    validate_policy(policy)
    # O perfil central não permite retirar perguntas da pontuação.
    if any(key not in {"audiences", "do_not_track"} for key in policy.get("applicability", {})):
        raise ValueError("core-14-v1 não permite substituir a aplicabilidade central.")
    scope = policy.get("document_scope")
    if scope is not None and scope not in {"full_document", "excerpt", "unknown"}:
        raise ValueError("Escopo inválido.")
    if policy["complete"] and scope in {"excerpt", "unknown"}:
        raise ValueError("complete contradiz document_scope.")
    extraction = policy.get("extraction", {})
    if not isinstance(extraction, dict) or extraction.get("status", "ok") not in {"ok", "partial"}:
        raise ValueError("Extração inválida.")
    raw = extraction.get("text")
    if raw is not None and not isinstance(raw, str):
        raise ValueError("Texto extraído deve ser string.")
    last = -1
    positioned = []
    for segment in policy["segments"]:
        page_start, page_end = segment.get('page_start'), segment.get('page_end')
        if not (page_start is None and page_end is None) and not (
                type(page_start) is int and type(page_end) is int and 1 <= page_start <= page_end):
            raise ValueError('Páginas exigem par de inteiros positivos ordenados ou ambos nulos.')
        a, b = segment.get("char_start"), segment.get("char_end")
        if a is None and b is None:
            continue
        if type(a) is not int or type(b) is not int or not 0 <= a < b:
            raise ValueError("Offsets globais inválidos.")
        if raw is not None and (b > len(raw) or raw[a:b] != segment["text"]):
            raise ValueError("Offsets globais não correspondem ao texto extraído.")
        positioned.append((a, b))
    for a, b in sorted(positioned):
        if a < last:
            raise ValueError("Segmentos com intervalos globais sobrepostos.")
        last = b


def _id(prefix: str, value: dict) -> str:
    return prefix + ":" + content_hash(value)


def _group(candidate: dict, question: dict, rules: heuristics.RuleSet, label: str) -> dict:
    window = candidate["window"]
    group = {"evidence": list(window.evidence), "polarity": candidate["polarity"],
             "detection_source": "model_and_rule" if label in window.labels else "rule_only",
             "rule_id": question["id"], "rule_version": rules.data["version"],
             "rationale": ("Relação contextual detectada." if candidate["state"] == "supported" else
                           "Informação pertinente; faltam: " + ", ".join(candidate["missing"]) if candidate["state"] == "partial" else
                           "Referência, exceção ou qualificador não resolvido; requer leitura humana.")}
    group["id"] = _id("evidence", {"rule": group["rule_id"], "version": group["rule_version"], "evidence": group["evidence"]})
    return group


def _finding(question: dict | None, rules: heuristics.RuleSet, kind: str, rationale: str,
             groups: list[dict], scope: dict | None, priority: str = "medium") -> dict:
    identity = {"rule": question["id"] if question else "input", "version": rules.data["version"],
                "kind": kind, "evidence": sorted(g["id"] for g in groups),
                "scope": content_hash(scope) if not groups else None, "reason": rationale}
    return {"id": _id("finding", identity), "dimension_id": question["id"].split('.')[0] if question else None,
            "question_id": question["id"] if question else None, "rule_id": identity["rule"],
            "rule_version": rules.data["version"], "kind": kind,
            "title": question["title"] if question else "Limites da entrada",
            "rationale": rationale, "priority": priority,
            "priority_reason": "Esclarecer declaração ou entrada impeditiva." if priority == "high" else "Conferir informação documental insuficiente ou ambígua.",
            "evidence_groups": groups, "search_scope": scope, "recommendation_ids": []}


def evaluate_policy(policy: dict, rules: heuristics.RuleSet, *, fallback: bool | None = None) -> dict:
    validate_input(policy)
    exclude_uncertain = rules.data['method_version'] in heuristics.UNCERTAINTY_METHODS
    fallback = rules.data["fallback"] if fallback is None else fallback
    if type(fallback) is not bool:
        raise ValueError("fallback deve ser booleano.")
    usable = [s for s in policy["segments"] if not s["truncated"]]
    excluded = sorted(s["segment_id"] for s in policy["segments"] if s["truncated"])
    contexts = heuristics.windows(policy, rules)
    dims = {d["id"]: d for d in rules.data["dimensions"]}
    questions = []
    findings = []
    feature_cache = {}
    for question in rules.data["questions"]:
        dimension = dims[question["id"].split('.')[0]]
        label = dimension["label"]
        selected = [w for w in contexts if fallback or label in w.labels]
        searched = sorted({s['segment_id'] for s in usable if fallback or label in s['labels']}
                          | {e['segment_id'] for w in selected for e in w.evidence})
        scope = {"searched_segment_ids": searched, "excluded_segment_ids": excluded,
                 "category_pass": True, "fallback_pass": fallback,
                 "fallback_reason": "Busca complementar em segmentos sem a categoria prevista." if fallback else None}
        candidates = []
        seen = set()
        occurrences = {}
        omitted = 0
        # Categoria primeiro, depois complemento; no desempate, conservar a primeira origem no documento.
        ordered = sorted(selected, key=lambda w: (label not in w.labels, w.order, w.joined))
        for window in ordered:
            candidate = heuristics.inspect(window, question, rules, feature_cache)
            if candidate:
                key = tuple((e['segment_id'], e['start'], e['end']) for e in window.evidence)
                if key not in seen:
                    seen.add(key)
                    statement = (re.sub(r'\s+', ' ', window.text).strip().casefold(), candidate['state'], candidate['polarity'])
                    occurrences[statement] = occurrences.get(statement, 0) + 1
                    if occurrences[statement] <= rules.data['max_evidence_occurrences']:
                        candidates.append(candidate)
                    else:
                        omitted += 1
        # Uma referência dependente não resolvida substitui a leitura isolada da frase precedente.
        ambiguous_joined = [c for c in candidates if c["window"].joined and c["state"] == "inconclusive"]
        if ambiguous_joined:
            candidates = [c for c in candidates if c in ambiguous_joined or not any(
                all(any(e["segment_id"] == x["segment_id"] and x["start"] <= e["start"] and e["end"] <= x["end"]
                        for x in joined["window"].evidence) for e in c["window"].evidence)
                for joined in ambiguous_joined)]
        # Repetições têm localizações próprias, mas só uma afirmação entra na busca de conflito.
        unique_statements = list({re.sub(r'\s+', ' ', c['window'].text).strip().casefold(): c for c in reversed(candidates)}.values())
        unique_statements.sort(key=lambda c: c['window'].order)
        conflicts = [(a, b) for a, b in combinations(unique_statements, 2) if heuristics.conflicting(a, b, question)]
        supporting = [c for c in candidates if c["state"] in {"supported", "partial"}]
        ambiguous = [c for c in candidates if c["state"] == "inconclusive"]
        context_candidates = ambiguous + [c for pair in conflicts for c in pair]
        if exclude_uncertain:
            # Também retirar janelas sobrepostas evita reintroduzir a mesma passagem duvidosa.
            uncertain_refs = [e for c in context_candidates for e in c['window'].evidence]
            uncertain_statements = {re.sub(r'\s+', ' ', c['window'].text).strip().casefold() for c in context_candidates}
            rejected = [c for c in supporting if re.sub(r'\s+', ' ', c['window'].text).strip().casefold() in uncertain_statements or any(
                e['segment_id'] == ref['segment_id'] and e['start'] < ref['end'] and ref['start'] < e['end']
                for e in c['window'].evidence for ref in uncertain_refs)]
            supporting = [c for c in supporting if c not in rejected]
            context_candidates += rejected
        if conflicts and not exclude_uncertain:
            state = "inconclusive"
        elif any(c["state"] == "supported" for c in supporting):
            state = "supported"
        elif supporting:
            state = "partial"
        elif context_candidates:
            state = "inconclusive"
        else:
            state = "not_detected"
        best = next((c for c in supporting if c["state"] == state), None)
        found = best["found"] if best else []
        missing = best["missing"] if best else list(question["elements"])
        rationale = {"supported": "Elementos exigidos encontrados em uma relação contextual.",
                     "partial": "Informação pertinente, mas incompleta: " + ", ".join(missing) + ".",
                     "not_detected": "Nenhum suporte localizado nos segmentos examinados; ausência não comprovada.",
                     "inconclusive": "Contexto ou declarações potencialmente incompatíveis impedem decidir."}[state]
        support_groups = [_group(c, question, rules, label) for c in supporting if exclude_uncertain or not conflicts]
        context_groups = list({g["id"]: g for g in (_group(c, question, rules, label) for c in context_candidates)}.values())
        local = []
        if state in {"partial", "not_detected"} and usable:
            local.append(_finding(question, rules, "information_gap", rationale, support_groups, scope))
        if ambiguous:
            local.append(_finding(question, rules, "ambiguity", "Referência ou exceção não resolvida.",
                                  [_group(c, question, rules, label) for c in ambiguous], scope))
        if conflicts:
            local.append(_finding(question, rules, "contradiction", "Declarações potencialmente incompatíveis sobre objeto/contexto comparáveis.", context_groups, scope, "high"))
        for candidate in candidates:
            if (dimension["id"] == "retention" and "indefinite" in candidate["features"]
                    or dimension["id"] in {"security", "dnt"} and candidate["polarity"] == "negated"):
                reason = ('Retenção explicitamente indefinida; revisar a prática declarada.' if dimension['id'] == 'retention'
                          else 'Resposta negativa ao sinal DNT; pontos documentais não comprovam proteção. Revisar a prática declarada.' if dimension['id'] == 'dnt'
                          else 'Negação de medida ou restrição de segurança; revisar a prática declarada.')
                local.append(_finding(question, rules, "practice_statement", reason,
                                      [_group(candidate, question, rules, label)], scope, "high"))
        local = list({f["id"]: f for f in local}.values())
        findings.extend(local)
        questions.append({"id": question["id"], "dimension_id": dimension["id"], "title": question["title"],
                          "status": state, "value": VALUES[state],
                          "applicability": {"state": "applicable", "source": "profile", "source_id": rules.data["profile_id"],
                                            "reason": "Pergunta documental central do perfil.", "declared_at": rules.data["profile_defined_at"]},
                          "rule_id": question["id"], "rule_version": rules.data["version"], "rationale": rationale,
                          "found_elements": found, "missing_elements": missing,
                          "supporting_groups": support_groups, "contextual_groups": context_groups,
                          "duplicate_occurrences_omitted": omitted,
                          "search_scope": scope, "finding_ids": [f["id"] for f in local]})
    blockers = []
    if not policy["complete"] or policy.get("document_scope") in {"excerpt", "unknown"}:
        blockers.append("document_scope_incomplete")
    if policy.get("extraction", {}).get("status") == "partial":
        blockers.append("extraction_partial")
    if excluded:
        blockers.append("truncated_segments")
    if not usable:
        blockers.append("no_usable_segments")
    input_partial = bool(blockers)
    if not exclude_uncertain and any(q["status"] == "inconclusive" for q in questions):
        blockers.append("inconclusive_questions")
    if exclude_uncertain and usable and all(q['value'] is None for q in questions):
        blockers.append('no_evaluable_questions')
    active = {q['dimension_id'] for q in questions if q['value'] is not None} if usable else set()
    total_weight = sum(d['weight'] for d in dims.values() if not exclude_uncertain or d['id'] in active)
    dimensions = []
    for dimension in dims.values():
        items = [q for q in questions if q["dimension_id"] == dimension["id"]]
        points = sum(q["value"] or 0 for q in items)
        resolved = sum(q["value"] is not None for q in items)
        denominator = resolved if exclude_uncertain else len(items)
        percentage = 100 * points / denominator if denominator and usable and (exclude_uncertain or resolved == len(items)) else None
        score = continuous_score(percentage) if percentage is not None else None
        weight = dimension['weight'] / total_weight if total_weight and (not exclude_uncertain or dimension['id'] in active) else 0.0
        dimensions.append({"id": dimension["id"], "title": dimension["title"], "weight": dimension["weight"],
                           "normalized_weight": weight, "status": "resolved" if score is not None else "unresolved",
                           "score": score, "contribution": (weight * score if exclude_uncertain else weight * 3 * points / len(items)) if not blockers and score is not None else None,
                           "earned_points": points, "applicable_questions": len(items), "unknown_applicability_questions": 0,
                           "resolved_questions": resolved, "inconclusive_questions": len(items) - resolved,
                           "excluded_questions": len(items) - resolved if exclude_uncertain else 0, "questions": items})
    if input_partial:
        findings.append(_finding(None, rules, "extraction_limitation", "Nota global bloqueada pelas condições de entrada: " + ", ".join(b for b in blockers if b != "inconclusive_questions"), [], None, "high"))
    return {"schema_version": 2, "policy_id": policy["policy_id"], "language": "pt",
            **({'uncertainty_policy': 'exclude_inconclusive_evidence_v1'} if exclude_uncertain else {}),
            "index": "documentary_information_quality", "method_version": rules.data["method_version"], "score_scale": SCORE_SCALE,
            "profile_id": rules.data["profile_id"], "profile_sha256": content_hash({"profile": rules.data["profile_id"], "dimensions": rules.data["dimensions"], "questions": rules.data["questions"]}),
            "validation_status": "not_human_validated", "input_status": "no_usable_segments" if not usable else "partial_input" if input_partial else "complete_input",
            "score_status": "blocked" if blockers else "available",
            "score": None if blockers else (min(3.0, sum(d['contribution'] or 0 for d in dimensions)) if exclude_uncertain else continuous_score(min(100, sum(
                100 * d['earned_points'] / d['applicable_questions'] * d['normalized_weight'] for d in dimensions)))),
            "score_blockers": blockers, "rules_version": rules.data["version"], "rules_sha256": content_hash(rules.data),
            "predictions_sha256": content_hash(policy), "fallback_enabled": fallback,
            "dimensions": dimensions, "findings": findings, "recommendations": [],
            "limitations": ["Avaliação documental experimental; não mede conformidade, proteção efetiva ou maturidade.",
                            "Heurísticas lexicais contextuais, sem compreensão semântica geral; revisão humana necessária.",
                            "Satisfação de uma pergunta indica ao menos uma relação encontrada, não um inventário exaustivo.",
                            "Conflitos reconhecidos são candidatos; variações de sujeito, escopo e exceções podem escapar.",
                            "Sem texto extraído e offsets, relações entre segmentos não são reconstruídas.",
                            "Catálogo de recomendações ainda não aplicado; nenhuma referência ISO foi inferida."]}


def render_report(report: dict) -> str:
    score = "não emitida" if report["score"] is None else f'{report["score"]}/3'
    lines = [f'# IQID experimental — {report["policy_id"]}', "", f'Nota global: **{score}**.',
             f'Método: `{report["method_version"]}`; regras: `{report["rules_version"]}`.',
             f'Bloqueios: {", ".join(report["score_blockers"]) or "nenhum"}.', "",
             "Notas de 0 a 3: até 25% → 0; acima de 25% até 50% → 1; acima de 50% até 75% → 2; acima de 75% → 3.",
             "A contribuição usa uma base contínua de 0 a 3 antes da classificação; não é uma nota.", "",
             "| Dimensão | Nota / 3 | Peso relativo | Contribuição |", "|---|---:|---:|---:|"]
    for dimension in report["dimensions"]:
        lines.append(f'| {dimension["title"]} | {dimension["score"]} | {dimension["normalized_weight"]:.6f} | {dimension["contribution"]} |')
    for dimension in report["dimensions"]:
        for question in dimension["questions"]:
            lines += ["", f'## {question["id"]}: {question["status"]}', "", question["rationale"]]
            if question['duplicate_occurrences_omitted']:
                lines.append(f"Ocorrências repetidas omitidas das citações: {question['duplicate_occurrences_omitted']} (sem efeito nos pontos).")
            for group in question["supporting_groups"] + question["contextual_groups"]:
                lines += ["", f'{group["detection_source"]}; {group["polarity"]}.']
                for evidence in group["evidence"]:
                    lines += [f'`{evidence["segment_id"]}` [{evidence["start"]}:{evidence["end"]}]',
                              *["> " + line for line in evidence["quote"].splitlines()]]
    lines += ["", "## Achados", ""]
    lines += [f'- {f["priority"]} — {f["title"]}: {f["rationale"]}' for f in report["findings"]]
    lines += ["", "## Limitações", "", *["- " + item for item in report["limitations"]]]
    return "\n".join(lines) + "\n"


def code_hashes() -> dict:
    import hashlib
    from . import scoring_common
    return {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (__file__, heuristics.__file__, scoring_common.__file__)}
