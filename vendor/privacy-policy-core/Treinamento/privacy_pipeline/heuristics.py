"""Regras textuais limitadas e rastreáveis; não é um analisador semântico geral."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
from typing import Any

import yaml

from .scoring_common import LABELS

DIMENSIONS = ("collection", "sharing", "choice", "access", "retention", "security", "changes")
QUESTION_IDS = {"collection.data_types", "collection.purposes", "sharing.recipients", "sharing.conditions",
                "choice.mechanism", "choice.channel", "access.channel", "access.response_time",
                "retention.duration", "retention.disposal", "security.technical_measures",
                "security.access_controls", "changes.notice_channel", "changes.notice_timing"}
VIRTUAL_FEATURES = {"negated"}


def expression(value: Any, facts: dict[str, bool]) -> bool:
    if isinstance(value, str):
        return facts[value]
    key, children = next(iter(value.items()))
    if key == "not":
        return not expression(children, facts)
    return (all if key == "all" else any)(expression(child, facts) for child in children)


def _validate_expression(value: Any, names: set[str]) -> None:
    if isinstance(value, str) and value in names:
        return
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("Expressão exige feature conhecida, all, any ou not.")
    key, children = next(iter(value.items()))
    if key == "not":
        _validate_expression(children, names)
    elif key in {"all", "any"} and isinstance(children, list) and children:
        for child in children:
            _validate_expression(child, names)
    else:
        raise ValueError("Operador composto inválido ou vazio.")


class RuleSet:
    """Valida e compila uma vez; o dicionário original identifica a configuração."""

    def __init__(self, rules: dict):
        if not isinstance(rules, dict):
            raise ValueError("Configuração de avaliação deve ser um objeto.")
        required = {"version", "method_version", "language", "profile_id", "profile_defined_at",
                    "max_sentences", "max_segments", "fallback", "features", "objects", "contexts",
                    "recipients", "dimensions", "questions", "max_evidence_occurrences"}
        if set(rules) != required:
            raise ValueError("Campos obrigatórios/inesperados na configuração de avaliação.")
        if rules["language"] != "pt" or rules["profile_id"] != "core-14-v1":
            raise ValueError("Somente pt e perfil core-14-v1 são suportados.")
        for key in ("version", "method_version", "profile_defined_at"):
            if not isinstance(rules[key], str) or not rules[key].strip():
                raise ValueError(f"{key} deve ser texto não vazio.")
        from datetime import datetime
        try:
            date = datetime.fromisoformat(rules["profile_defined_at"])
            if date.utcoffset() is None:
                raise ValueError()
        except ValueError:
            raise ValueError("profile_defined_at exige data ISO-8601 com fuso.") from None
        if (type(rules["fallback"]) is not bool or type(rules["max_sentences"]) is not int
                or type(rules["max_segments"]) is not int or rules["max_sentences"] not in (1, 2)
                or rules["max_segments"] not in (1, 2)):
            raise ValueError("Janela limitada a uma/duas frases e segmentos; fallback booleano.")
        if type(rules['max_evidence_occurrences']) is not int or not 2 <= rules['max_evidence_occurrences'] <= 10:
            raise ValueError('Limite de citações repetidas deve estar entre 2 e 10.')
        self.patterns: dict[str, dict[str, re.Pattern]] = {}
        for group in ("features", "objects", "contexts", "recipients"):
            entries = rules[group]
            if not isinstance(entries, dict) or not entries:
                raise ValueError(f"Dicionário {group} vazio ou inválido.")
            self.patterns[group] = {}
            for name, pattern in entries.items():
                if not isinstance(name, str) or not isinstance(pattern, str) or not pattern:
                    raise ValueError("Feature e regex devem ser textos não vazios.")
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                except re.error as error:
                    raise ValueError(f"Regex inválida em {group}.{name}: {error}") from error
                if compiled.search(""):
                    raise ValueError("Regex não pode aceitar texto vazio.")
                self.patterns[group][name] = compiled
        dims = rules["dimensions"]
        if not {"continuation", "request", "change", "unresolved", "dispose", "indefinite"} <= set(rules["features"]):
            raise ValueError("Faltam features estruturais de contexto.")
        if set(rules['features']) & VIRTUAL_FEATURES:
            raise ValueError("Features virtuais são reservadas ao motor.")
        if not isinstance(dims, list) or [d.get("id") for d in dims if isinstance(d, dict)] != list(DIMENSIONS):
            raise ValueError("Configure as sete dimensões centrais na ordem do perfil.")
        for dim in dims:
            if set(dim) != {"id", "title", "weight", "label"} or dim["label"] not in LABELS:
                raise ValueError("Campos/categoria de dimensão inválidos.")
            if dim['label'] != LABELS[DIMENSIONS.index(dim['id'])] or not isinstance(dim['title'], str) or not dim['title'].strip():
                raise ValueError("Título ou categoria incompatível com o perfil central.")
            weight = dim["weight"]
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                raise ValueError("Peso deve ser positivo e finito.")
        if not math.isfinite(sum(d["weight"] for d in dims)):
            raise ValueError("Soma de pesos não finita.")
        questions = rules["questions"]
        if not isinstance(questions, list) or len(questions) != 14:
            raise ValueError("O perfil exige 14 perguntas.")
        names = set(rules["features"]) | VIRTUAL_FEATURES
        ids = []
        for question in questions:
            if not isinstance(question, dict) or set(question) != {"id", "title", "predicate", "candidate", "elements", "conflict"}:
                raise ValueError("Campos de pergunta inválidos.")
            if any(not isinstance(question[key], str) or not question[key].strip() for key in ("id", "title", "predicate")):
                raise ValueError("Pergunta exige id, título e predicado.")
            ids.append(question["id"])
            if question["predicate"] not in rules["features"] or question["conflict"] not in {"polarity", "duration", "disposal", "timing"}:
                raise ValueError("Predicado ou conflito desconhecido.")
            _validate_expression(question["candidate"], names)
            if not isinstance(question["elements"], dict) or not question["elements"]:
                raise ValueError("Pergunta sem elementos exigidos.")
            for value in question["elements"].values():
                _validate_expression(value, names)
        if set(ids) != QUESTION_IDS:
            raise ValueError("IDs duplicados ou distribuição de perguntas inválida.")
        self.data = rules

    def hits(self, group: str, text: str) -> set[str]:
        return {name for name, pattern in self.patterns[group].items() if pattern.search(text)}


def load_rules(path: str | Path) -> RuleSet:
    return RuleSet(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Window:
    text: str
    evidence: tuple[dict, ...]
    labels: frozenset[str]
    order: tuple[int, int]
    joined: bool = False
    previous_text: str = ""


def _evidence(segment: dict, start: int, end: int) -> dict:
    return {"segment_id": segment["segment_id"], "start": start, "end": end, "quote": segment["text"][start:end]}


def windows(policy: dict, rules: RuleSet) -> list[Window]:
    """Divide sem reescrever; só cruza segmentos com continuidade comprovada."""
    segments = policy["segments"]
    if segments and all("char_start" in s for s in segments):
        segments = sorted(segments, key=lambda s: s["char_start"])
    raw = policy.get("extraction", {}).get("text")
    # Blocos mantêm o mapa de cada fragmento para o segmento original.
    blocks: list[list[tuple[dict, int]]] = []
    for index, segment in enumerate(segments):
        if segment["truncated"]:
            continue
        blocks.append([(segment, 0)])
        previous = segments[index - 1] if index and not segments[index - 1]['truncated'] else None
        gap = None
        if previous and raw is not None and "char_end" in previous and "char_start" in segment:
            gap = raw[previous["char_end"]:segment["char_start"]]
        same_page = previous is not None and previous.get("page_end") == segment.get("page_start")
        if (gap is not None and same_page and not gap.strip() and not re.search(r"\n\s*\n", gap)
                and rules.data["max_segments"] == 2):
            blocks.append([(previous, 0), (segment, len(previous['text']) + len(gap))])
    result = []
    by_interval = {}
    for block_index, block in enumerate(blocks):
        if not block:
            continue
        if len(block) == 1:
            text = block[0][0]["text"]
        else:
            text = raw[block[0][0]["char_start"]:block[-1][0]["char_end"]]
        for paragraph in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
            spans = []
            left = paragraph.start()
            for boundary in re.finditer(r"[.!?;](?=\s|$)|,\s*(?=(?:mas|porém|contudo|enquanto|e\s+para)\b)", paragraph.group(), re.I):
                right = paragraph.start() + boundary.end()
                if text[left:right].strip():
                    spans.append((left, right))
                left = right
            if text[left:paragraph.end()].strip():
                spans.append((left, paragraph.end()))
            expanded = [(a, b, False, text[spans[i-1][0]:spans[i-1][1]].strip() if i else '')
                        for i, (a, b) in enumerate(spans)]
            if rules.data["max_sentences"] == 2:
                for (a, b), (c, d) in zip(spans, spans[1:]):
                    following = text[c:d].strip()
                    # Duas frases não bastam: exigir ligação anafórica explícita.
                    if rules.patterns["features"]["continuation"].search(following):
                        expanded.append((a, d, True, ''))
            for start, end, joined, previous_text in expanded:
                while start < end and text[start].isspace():
                    start += 1
                refs = []
                labels = set()
                for segment, offset in block:
                    a, b = max(start - offset, 0), min(end - offset, len(segment["text"]))
                    if a < b:
                        refs.append(_evidence(segment, a, b))
                        labels.update(segment["labels"])
                # No bloco sobreposto, conservar apenas janelas que cruzem a fronteira.
                if len(block) == 2 and len(refs) < 2:
                    # A frase isolada pode usar a anterior para comparar objetos referidos
                    # ("esses mesmos dados"). O par conflitante citará ambas as frases.
                    key = tuple((e['segment_id'], e['start'], e['end']) for e in refs)
                    if previous_text and key in by_interval:
                        index = by_interval[key]
                        if not result[index].previous_text:
                            result[index] = replace(result[index], previous_text=previous_text)
                    continue
                key = tuple((e['segment_id'], e['start'], e['end']) for e in refs)
                by_interval[key] = len(result)
                result.append(Window(text[start:end], tuple(refs), frozenset(labels), (block_index, start), joined, previous_text))
    return result


def polarity(text: str, predicate: re.Pattern) -> str:
    match = predicate.search(text)
    prefix = text[:match.start()] if match else ""
    if re.search(r"\bnão\s+(?:só|apenas|somente)\b", text, re.I):
        return "uncertain"
    # A negação precisa alcançar a ação: "não controla o compartilhamento"
    # nega o controle, não a prática de compartilhar.
    if (re.search(r"\b(?:não|nunca|jamais|nem)\s+(?:(?:é|são|será|serão|foi|foram|"
                  r"pode|podem|podemos|poderá|poderão|vamos|iremos|possível|"
                  r"intencionalmente|conscientemente|automaticamente)\s+){0,3}$", prefix, re.I)
            or re.search(r"\bsem restrição\b", text, re.I)):
        return "negated"
    if re.search(r"(?:^|[,;])\s*(?:se|caso)\b|\b(?:se|caso)\s+(?:você|houver|for|necessário)\b|\b(?:poderemos|podemos|poderá)\b", text, re.I):
        return "conditional"
    return "affirmed"


def inspect(window: Window, question: dict, rules: RuleSet, cache: dict | None = None) -> dict | None:
    text = window.text
    def hits(group, value):
        key = (group, value)
        if cache is None:
            return rules.hits(group, value)
        if key not in cache:
            cache[key] = rules.hits(group, value)
        return cache[key]
    found = hits("features", text)
    if 'non_assertion' in found:
        return None
    mode = polarity(text, rules.patterns["features"][question["predicate"]])
    if question['id'] == 'security.technical_measures':
        for measure in rules.patterns['features']['measure'].finditer(text):
            if re.search(r'\bsem\s+(?:\w+\s+){0,2}$', text[:measure.start()], re.I):
                mode = 'negated'
    facts = {name: name in found for name in rules.data["features"]}
    facts.update(negated=mode == "negated")
    if not expression(question["candidate"], facts):
        return None
    elements = {name: expression(value, facts) for name, value in question["elements"].items()}
    uncertain = "unresolved" in found or mode == "uncertain"
    # Exceções abertas e referências pendentes não são preenchidas por proximidade.
    if re.search(r"\bexceto\b", text, re.I) and not re.search(r"\b(?:por obrigação legal|ordem judicial)\b", text, re.I):
        uncertain = True
    if mode == 'conditional' and re.search(r'\b(?:se necessário|se for possível|caso necessário)\b', text, re.I):
        uncertain = True
    if (window.joined and question['id'] == 'changes.notice_timing'
            and re.search(r'\bmesmo aviso\b', text, re.I)
            and re.search(r'\bantes\b', text, re.I) and re.search(r'\b(?:depois|após)\b', text, re.I)):
        uncertain = True
    state = "inconclusive" if uncertain else "supported" if all(elements.values()) else "partial"
    return {"window": window, "state": state, "polarity": "uncertain" if uncertain else mode,
            "found": sorted(name for name, ok in elements.items() if ok),
            "missing": sorted(name for name, ok in elements.items() if not ok),
            "objects": hits("objects", text), "contexts": hits("contexts", text),
            "recipients": hits("recipients", text), "features": found}


def conflicting(a: dict, b: dict, question: dict) -> bool:
    """Sinais conservadores de conflito; requerem objeto/contexto comparáveis."""
    wa, wb = a["window"], b["window"]
    # Uma janela ampliada e sua própria frase não são duas declarações.
    if any(x["segment_id"] == y["segment_id"] and max(x["start"], y["start"]) < min(x["end"], y["end"])
           for x in wa.evidence for y in wb.evidence):
        return False
    explicit_same = (wb.previous_text == wa.text and bool(re.search(
        r"\b(?:esses? mesmos?|essas? mesmas?|mesm[oa] (?:conta|pedido|aviso))\b", wb.text, re.I)))
    if not (a["objects"] and a['objects'] == b['objects'] or (explicit_same and a["objects"])):
        return False
    contexts_a, contexts_b = a['contexts'], b['contexts']
    if question['id'] == 'security.access_controls':
        # Ator é precisamente o objeto da divergência nessa pergunta.
        contexts_a, contexts_b = contexts_a - {'employees'}, contexts_b - {'employees'}
    if contexts_a != contexts_b or a["recipients"] != b["recipients"]:
        return False
    if set(re.findall(r'\b20\d{2}\b', wa.text)) != set(re.findall(r'\b20\d{2}\b', wb.text)):
        return False
    # Condições ou exceções não resolvidas impedem afirmar equivalência de contexto.
    if a["polarity"] in {"conditional", "uncertain"} or b["polarity"] in {"conditional", "uncertain"}:
        return False
    kind = question["conflict"]
    if kind == "polarity":
        return {a["polarity"], b["polarity"]} == {"affirmed", "negated"}
    if kind == "disposal":
        # Direito de excluir a qualquer momento não contradiz retenção enquanto
        # a conta está ativa; comparar destinações exige um evento em ambas.
        if 'end_event' not in a['features'] or ('end_event' not in b['features'] and not explicit_same):
            return False
        return ("dispose" in a["features"]) != ("dispose" in b["features"])
    if kind == "duration":
        def periods(value: dict) -> set[str]:
            return {m.group().lower() for m in re.finditer(r"\b\d+\s+(?:dias?|meses|anos?|horas?)\b|\bindefinidamente\b", value["window"].text, re.I)}
        pa, pb = periods(a), periods(b)
        return bool(pa and pb and pa != pb)
    return bool((re.search(r"\bantes\b", wa.text, re.I) and re.search(r"\b(?:depois|após)\b", wb.text, re.I))
                or (re.search(r"\bantes\b", wb.text, re.I) and re.search(r"\b(?:depois|após)\b", wa.text, re.I)))
