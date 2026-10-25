"""Planilha simples e estável para intercâmbio com o Google Planilhas."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def write_workbook(frame: pd.DataFrame, path: Path, sheet_name: str = "trechos") -> None:
    """Exporta valores literais, sem fórmulas e sem truncamento silencioso."""
    book = Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = sheet_name
    sheet.append(list(frame.columns))
    for values in frame.itertuples(index=False, name=None):
        if any(isinstance(value, str) and len(value) > 32767 for value in values):
            raise ValueError("Texto ultrapassa limite de 32767 caracteres do Excel.")
        sheet.append(list(values))
        for cell in sheet[sheet.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    for cell in sheet[1]:
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="23364D")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[1].height = 64
    widths = {"segment_id": 16, "policy_id": 12, "texto_en": 100, "texto_pt": 100,
              "labels": 50, "n_labels": 10, "split": 14}
    for index, name in enumerate(frame.columns, 1):
        sheet.column_dimensions[get_column_letter(index)].width = widths.get(name, 22)
    for row in sheet.iter_rows(min_row=2):
        line_count = 1
        for cell in row:
            name = str(sheet.cell(1, cell.column).value)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if name == "texto_pt":
                cell.fill = PatternFill("solid", fgColor="FFF2CC")
            if name in {"texto_en", "texto_pt", "labels"} and cell.value:
                line_count = max(line_count, sum(max(1, len(textwrap.wrap(line, int(widths[name]) - 8))) for line in str(cell.value).splitlines()))
        sheet.row_dimensions[row[0].row].height = min(409, max(30, math.ceil(line_count * 15 + 8)))
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)


def verify_workbook(path: Path, expected: pd.DataFrame) -> None:
    """Relê cada célula para detectar perdas, fórmulas indevidas e IDs alterados."""
    book = load_workbook(path, read_only=True, data_only=False)
    rows = list(book.active.iter_rows())
    if [cell.value for cell in rows[0]] != list(expected.columns):
        raise AssertionError("Cabeçalhos divergentes após a exportação.")
    if len(rows) != len(expected) + 1:
        raise AssertionError("Número de linhas divergente.")
    for cells, values in zip(rows[1:], expected.itertuples(index=False, name=None), strict=True):
        for cell, value in zip(cells, values, strict=True):
            if cell.data_type in {"f", "e"} or (cell.value if cell.value is not None else "") != value:
                raise AssertionError(f"Valor divergente em {cell.coordinate}.")
    book.close()
