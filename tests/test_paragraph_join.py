"""Frases quebradas por linhas em branco do PDF voltam a ser um parágrafo só."""
import pytest

from privacy_service.extraction import extract_document
from privacy_service.segmentation import paragraph_ranges


def paragraphs(text):
    return [text[a:b] for a, b in paragraph_ranges(text, 0, len(text))]


@pytest.mark.parametrize('text,expected', [
    # PDF com espaço duplo: cada linha virava um trecho.
    ('A UNIP elaborou este documento para\n\norientar e esclarecer sobre as práticas.',
     ['A UNIP elaborou este documento para\n\norientar e esclarecer sobre as práticas.']),
    # Lista introduzida por dois-pontos.
    ('Você poderá compreender:\n\ncomo a CAIXA obtém dados;\n\nquais dados a CAIXA trata.',
     ['Você poderá compreender:\n\ncomo a CAIXA obtém dados;\n\nquais dados a CAIXA trata.']),
    # Linha que termina em palavra de ligação ou vírgula continua, mesmo se a próxima tem maiúscula.
    ('Cada serviço pode ter uma\n\nPolítica de Privacidade própria.', ['Cada serviço pode ter uma\n\nPolítica de Privacidade própria.']),
    ('Nesta Política de Privacidade,\n\n"Afiliadas" são nossas subsidiárias.', ['Nesta Política de Privacidade,\n\n"Afiliadas" são nossas subsidiárias.']),
    # Frase terminada: o próximo parágrafo continua separado, mesmo em minúscula.
    ('Coletamos seu nome.\n\nem seguida, usamos o e-mail.', ['Coletamos seu nome.', 'em seguida, usamos o e-mail.']),
    # Novo parágrafo em maiúscula continua separado.
    ('Coleta de dados\n\nColetamos seu nome.', ['Coleta de dados', 'Coletamos seu nome.']),
    # Título e data isolados não se juntam, para o filtro de não conteúdo seguir funcionando.
    ('Alterações nesta política\n\noutubro de 2026\n\nColetamos seu nome.',
     ['Alterações nesta política', 'outubro de 2026', 'Coletamos seu nome.']),
])
def test_paragraph_ranges(text, expected):
    assert paragraphs(text) == expected


def test_prepare_keeps_offsets_of_joined_paragraph(segmenter):
    text = 'Elaboramos esta Política para\n\norientar os titulares sobre a coleta de dados.\n\nColetamos seu nome.'
    prepared = segmenter.prepare(extract_document(text.encode('utf-8'), 'politica.txt', document_scope='full_document'))
    assert [s['text'] for s in prepared['segments']] == [
        'Elaboramos esta Política para\n\norientar os titulares sobre a coleta de dados.', 'Coletamos seu nome.']
    for segment in prepared['segments']:
        assert prepared['extraction']['text'][segment['char_start']:segment['char_end']] == segment['text']
