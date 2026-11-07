"""Valida a escolha de temas; não altera a integridade declarada do documento."""
from .heuristics import PROFILES

CURRENT_PROFILE = 'documentary-18-v1'


def resolve_assessment_scope(value=None):
    dimensions = list(PROFILES[CURRENT_PROFILE][0])
    if value is None:
        return {'mode': 'all_categories', 'category_ids': dimensions}
    if not isinstance(value, dict) or set(value) != {'mode', 'category_ids'}:
        raise ValueError('Informe o modo e as categorias a avaliar.')
    ids = value['category_ids']
    if (value['mode'] not in {'all_categories', 'selected_categories'} or not isinstance(ids, list)
            or not ids or any(not isinstance(i, str) or i not in dimensions for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Selecione ao menos uma categoria válida, sem repetições.')
    if value['mode'] == 'all_categories' and set(ids) != set(dimensions):
        raise ValueError('O modo Todas exige as nove categorias do perfil.')
    return {'mode': value['mode'], 'category_ids': [i for i in dimensions if i in ids]}
