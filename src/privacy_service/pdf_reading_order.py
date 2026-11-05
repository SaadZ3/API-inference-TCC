"""Leitura por posição, com células separadas e conferência contra o texto original."""
from collections import Counter
import unicodedata


def characters(text: str) -> Counter:
    return Counter(c for c in unicodedata.normalize('NFKC', text) if not c.isspace())


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
    events = [(line['top'], line['x0'], line['bottom'], line['text'], False)
              for line in body.extract_text_lines(x_tolerance=2, y_tolerance=3, expand_ligatures=False, return_chars=False)]
    for cell in cells:
        part = clean.filter(lambda obj: obj['object_type'] == 'char' and inside(obj, cell))
        text = part.extract_text(x_tolerance=2, y_tolerance=3, expand_ligatures=False) or ''
        if text.strip():
            events.append((cell[1], cell[0], cell[3], text, True))
    result = []
    previous = None
    for top, left, bottom, text, table_cell in sorted(events):
        if previous is not None:
            previous_bottom, previous_height, previous_cell = previous
            paragraph = table_cell or previous_cell or top - previous_bottom > max(3, previous_height * 0.8)
            result.append('\n\n' if paragraph else '\n')
        result.append(text)
        previous = bottom, bottom - top, table_cell
    text = ''.join(result)
    expected = ''.join(char['text'] for char in clean.chars)
    if characters(text) != characters(expected):
        raise ValueError('A organização das células alterou o conteúdo extraído.')
    return text
