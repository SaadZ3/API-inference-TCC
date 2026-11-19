"""Leitura por posição, com células separadas e conferência contra o texto original."""
from collections import Counter
import unicodedata


def characters(text: str) -> Counter:
    return Counter(c for c in unicodedata.normalize('NFKC', text) if not c.isspace())


COLUMN_TOLERANCE = 4


def split_at_wide_gaps(line: dict) -> list[dict]:
    """Corta a linha onde o vão entre letras é grande demais para ser um espaço.

    Ex.: "Por quanto tempo mantemos suas      aplicativos, sites..." vira dois pedaços,
    porque o primeiro é um item do menu lateral e o segundo é o corpo do texto.
    """
    chars, text = line['chars'], line['text']
    whole = [{'top': line['top'], 'bottom': line['bottom'], 'x0': line['x0'], 'x1': line['x1'],
              'text': text, 'after_gap': False}]
    positions, index = [], 0
    for offset, char in enumerate(text):
        if index < len(chars) and chars[index]['text'] == char:
            positions.append(offset)
            index += 1
        elif not char.isspace():
            return whole
    if index != len(chars):
        return whole
    cuts = [i for i in range(1, len(chars))
            if chars[i]['x0'] - chars[i - 1]['x1'] > max(30.0, 2.5 * chars[i - 1]['size'])]
    if not cuts:
        return whole
    pieces = []
    for start, end in zip([0] + cuts, cuts + [len(chars)]):
        part = chars[start:end]
        pieces.append({'top': min(c['top'] for c in part), 'bottom': max(c['bottom'] for c in part),
                       'x0': min(c['x0'] for c in part), 'x1': max(c['x1'] for c in part),
                       'text': text[positions[start]:positions[end - 1] + 1].strip(), 'after_gap': start > 0})
    return pieces


def column_zones(events: list[dict]) -> list[tuple[int, int, float]]:
    """Acha faixas verticais com um corredor vazio entre duas colunas.

    Um x vira fronteira quando 2+ pedaços cortados começam nele. A zona cresce para
    cima e para baixo enquanto nenhuma linha atravessa esse x; uma que atravesse
    (texto de largura total ou célula de tabela) encerra a zona.
    """
    def crosses(event, boundary):
        return event['cell'] or (event['x0'] < boundary - COLUMN_TOLERANCE
                                 and event['x1'] > boundary + COLUMN_TOLERANCE)

    starts = [e['x0'] for e in events if e['after_gap']]
    candidates = sorted({s for s in starts if sum(abs(o - s) <= COLUMN_TOLERANCE for o in starts) >= 2},
                        key=lambda s: -sum(abs(o - s) <= COLUMN_TOLERANCE for o in starts))
    zones, used = [], set()
    for boundary in candidates:
        hits = [i for i, e in enumerate(events) if e['after_gap'] and abs(e['x0'] - boundary) <= COLUMN_TOLERANCE]
        first, last = min(hits), max(hits)
        if any(crosses(events[i], boundary) for i in range(first, last + 1)) or used & set(range(first, last + 1)):
            continue
        while first > 0 and not crosses(events[first - 1], boundary) and first - 1 not in used:
            first -= 1
        while last + 1 < len(events) and not crosses(events[last + 1], boundary) and last + 1 not in used:
            last += 1
        zones.append((first, last, boundary))
        used |= set(range(first, last + 1))
    return sorted(zones)


def order_columns(events: list[dict]) -> list[dict]:
    """Dentro de cada zona de colunas, entrega a coluna esquerda inteira e depois a direita."""
    ordered, cursor = [], 0
    for first, last, boundary in column_zones(events):
        ordered.extend(events[cursor:first])
        zone = events[first:last + 1]
        left = [e for e in zone if e['x0'] < boundary - COLUMN_TOLERANCE]
        right = [e for e in zone if e['x0'] >= boundary - COLUMN_TOLERANCE]
        ordered.extend(left)
        # Só a volta ao topo, na coluna da direita, precisa de quebra forçada; o resto
        # segue a regra normal de distância entre linhas.
        ordered.extend({**e, 'stream_start': i == 0 and bool(left)} for i, e in enumerate(right))
        cursor = last + 1
    ordered.extend(events[cursor:])
    return ordered


def read_positioned_page(page, original: str) -> str:
    # A conferência acontece antes de remover letras desenhadas duas vezes no mesmo lugar.
    spatial = page.extract_text(x_tolerance=2, y_tolerance=3, expand_ligatures=False) or ''
    if characters(spatial) != characters(original):
        raise ValueError('As duas leituras não preservam os mesmos caracteres.')
    # Um conjunto evita a busca quadrática da biblioteca em páginas muito longas.
    # Só elimina desenhos sobrepostos; frases repetidas em outra posição permanecem.
    seen = set()
    unique_ids = set()
    for char in page.chars:
        key = (char['text'], char['fontname'], char['size'], char['upright'],
               round(char['x0'], 3), round(char['top'], 3))
        if key not in seen:
            seen.add(key)
            unique_ids.add(id(char))
    clean = page.filter(lambda obj: obj['object_type'] != 'char' or id(obj) in unique_ids)
    # Bordas da página e cartões de uma coluna não são tabelas de dados.
    tables = [table for table in clean.find_tables() if len({round(cell[0], 2) for cell in table.cells if cell}) > 1]
    cells = sorted({cell for table in tables for cell in table.cells if cell is not None}, key=lambda c: (c[1], c[0]))
    cells = [cell for cell in cells if not any(
        other != cell and cell[0] <= other[0] and cell[1] <= other[1]
        and cell[2] >= other[2] and cell[3] >= other[3] for other in cells)]

    def inside(obj, box):
        x = (obj['x0'] + obj['x1']) / 2
        y = (obj['top'] + obj['bottom']) / 2
        return box[0] <= x < box[2] and box[1] <= y < box[3]

    body = clean.filter(lambda obj: obj['object_type'] != 'char' or not any(inside(obj, cell) for cell in cells))
    lines = body.extract_text_lines(x_tolerance=2, y_tolerance=3, expand_ligatures=False, return_chars=True)
    events = [{**piece, 'cell': False} for line in lines for piece in split_at_wide_gaps(line)]
    for cell in cells:
        part = clean.filter(lambda obj: obj['object_type'] == 'char' and inside(obj, cell))
        text = part.extract_text(x_tolerance=2, y_tolerance=3, expand_ligatures=False) or ''
        if text.strip():
            events.append({'top': cell[1], 'x0': cell[0], 'bottom': cell[3], 'x1': cell[2],
                           'text': text, 'cell': True, 'after_gap': False})
    events.sort(key=lambda e: (e['top'], e['x0'], e['bottom']))
    events = order_columns(events)
    if not any(e.get('stream_start') for e in events):
        # Sem coluna comprovada, a leitura fica exatamente como antes: linhas inteiras.
        events = [{'top': l['top'], 'x0': l['x0'], 'bottom': l['bottom'], 'text': l['text'], 'cell': False}
                  for l in lines] + [e for e in events if e['cell']]
        events.sort(key=lambda e: (e['top'], e['x0'], e['bottom']))
    result = []
    previous = None
    for event in events:
        top, bottom = event['top'], event['bottom']
        if previous is not None:
            previous_bottom, previous_height, previous_cell = previous
            paragraph = (event.get('stream_start') or event['cell'] or previous_cell
                         or top - previous_bottom > max(3, previous_height * 0.8))
            result.append('\n\n' if paragraph else '\n')
        result.append(event['text'])
        previous = bottom, bottom - top, event['cell']
    text = ''.join(result)
    expected = ''.join(char['text'] for char in clean.chars)
    if characters(text) != characters(expected):
        raise ValueError('A organização das células alterou o conteúdo extraído.')
    return text
