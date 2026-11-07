# Serviço Python de análise

**Atual R10:** perfil `documentary-18-v1`, nove categorias e 18 perguntas, mantendo a R9 para excluir dúvidas. Histórico de sete categorias preservado. Exige a migration `20261106000100_documentary_18_profile` e atualização coordenada dos três serviços. Implementação local, sem publicação; 7/8 exemplos adicionais corresponderam à expectativa. Ver [entrega e limites](../docs/R10_IMPLEMENTACAO.md). As marcações R9/R8/R7 abaixo são históricas.

**Atual R9:** método `documentary-quality-v4-draft1` e regras `governance-rules-v4-dev1`. Dúvidas locais ficam fora da pontuação e não bloqueiam o total. Ver [regra](../docs/R9_CALCULO_SEM_BLOQUEIO_POR_DUVIDAS.md) e [testes](../docs/R9_IMPLEMENTACAO.md). Backend e frontend devem ser atualizados juntos; esta tarefa não publicou serviços. O registro R7/R8 abaixo é histórico.

**R7/R8:** regras locais `governance-rules-v3-dev2`, congeladas para avaliação. Publicação suspensa por falhas de reconhecimento documentadas na [R7](../docs/R7_REVALIDACAO_NOTAS.md). Persistência e entrega local conferidas na [R8](../docs/R8_CONSOLIDACAO_ENTREGA.md). Não interpretar as instruções de publicação abaixo como aceite desta versão.

R6: novas avaliações usam `continuous_0_3` e método `documentary-quality-v3-draft1`. Backend e frontend precisam aceitar a escala contínua antes da publicação. O perfil mantém 14 perguntas, os bloqueios de entrada e as recomendações. Ver [relatório](../docs/R6_IMPLEMENTACAO_NOTAS.md).

Para publicar pelo site do Railway, siga [RAILWAY.md](RAILWAY.md). O repositório agora inclui o núcleo em `vendor/`, regras em `config/` e metadados em `model/`. Os pesos grandes são distribuídos como anexo de Release.

O projeto usa um único método de avaliação da qualidade da informação documentada, com notas de **0 a 3**. Consulte [a atualização do método](../docs/AVALIACAO_UNIFICADA.md).

P4 disponível: PDF digital/TXT UTF-8 → extração → BERTimbau → score com evidências. Serviço interno, independente do banco. O navegador acessa o serviço por meio do backend Node, já integrado ao fluxo de análise.

## Instalar e iniciar

Na pasta tcc-inference, em PowerShell:

```powershell
# Apenas se o ambiente ainda não existir:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m uvicorn privacy_service.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Se .venv já existe, comece pelo pip install para apontar o núcleo para a cópia em `vendor/`. Dependências fixadas em requirements.lock.txt, incluindo PyTorch. Não há re-treinamento. Os pesos devem existir em `model/`; na hospedagem, `scripts/start_railway.py` baixa o anexo da Release antes de iniciar. Ambiente validado originalmente: Windows, Python 3.13.7 e CPU.

O .env é opcional com a disposição atual de pastas. Consulte .env.example para MODEL_DIR, SCORING_RULES, DEVICE=cpu, BATCH_SIZE=4, CPU_THREADS=4, MAX_UPLOAD_BYTES=10485760, ANALYSIS_TIMEOUT_SECONDS=120 e UPLOAD_TIMEOUT_SECONDS=30. Caminhos relativos da API são resolvidos a partir de tcc-inference; variáveis do processo têm prioridade. Não sobrescreva seu .env para atualizar.

Use **um worker**, sem --reload na execução normal: cada processo carrega uma cópia do modelo. Não exponha esta API diretamente na internet.

## Conferir e analisar

- GET /health/live: processo responde.
- GET /health/ready: 200 somente após conferir hashes, carregar modelo/regras e executar aquecimento real; caso contrário, 503.
- GET /manifest: versões, hashes e configuração usada, sem caminhos locais.
- POST /analyze: arquivo multipart; resposta com schema_version, versions e result.
- /docs: formulário interativo para testar o envio.

Em outro terminal, ainda em tcc-inference:

```powershell
curl.exe --fail-with-body http://127.0.0.1:8000/health/ready
New-Item -ItemType Directory -Force outputs | Out-Null
curl.exe --fail-with-body --max-time 160 -F "file=@examples/politica_sintetica.txt" -F "language=pt" -F "document_scope=full_document" http://127.0.0.1:8000/analyze -o outputs/politica_sintetica_p4.json
```

O último comando grava o resultado; use outro nome de saída para preservar uma análise anterior. Para arquivo próprio: -F "file=@C:/caminho/politica.pdf". Escolha full_document (política inteira), excerpt (trecho) ou unknown (escopo desconhecido). Excerpt/unknown e extrações parciais têm score global nulo; as notas locais disponíveis descrevem somente o texto analisado.

Limites: 10 MiB, 50 páginas, 100 mil caracteres, 30 s para upload e 120 s para processamento. Uma análise por vez; concorrência retorna 503, sem fila. O timeout HTTP não interrompe uma operação nativa em andamento: o serviço permanece ocupado até ela terminar e verifica o prazo entre etapas/lotes. Se uma operação travar, reinicie o processo. Detalhes e medidas em [P4](../docs/P4.md).

## Preparar sem classificar — CLI da P3

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
.\.venv\Scripts\python.exe -m privacy_service.prepare --input examples/politica_sintetica.txt --output outputs/politica_preparada.json --model-dir "model" --document-scope full_document
```

A CLI não carrega pesos e recusa sobrescrever arquivos. Seus caminhos relativos são interpretados a partir do terminal. PDF escaneado precisa de OCR, fora do protótipo. Ver [P3](../docs/P3.md).

## Testar

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes usuais usam o tokenizer real, regras e substitutos controlados do classificador para testar HTTP e falhas; não carregam pesos. TCC_TEST_MODEL_DIR permite alterar o caminho do pacote.

Validação adicional com pesos reais, equivalência com a rodada anterior e medidas de memória/tempo:

```powershell
.\.venv\Scripts\python.exe scripts/validate_real.py --model-dir "model" --rules "config/scoring.yaml" --pdf "C:/caminho/politica.pdf" --output-dir verification/p4_nova
```

Use uma pasta de saída nova. O script processa seu PDF inteiro e como trecho, testa TXT sintético de 100 mil caracteres, compara 32 trechos históricos e salva respostas/manifesto/medidas. A comparação pressupõe que os outputs históricos originais ainda existem.

Para conferir mudanças nas regras sem executar novamente o classificador, use `scripts/revalidate_exports.py --inputs <pasta> --output <nova-pasta>`. A entrada deve conter PDFs e exportações `resultado-*.json`; o script associa os arquivos por hash, reextrai o texto e recusa reutilizar classificações se o texto mudar. As saídas identificam a análise original e não são gravadas no banco. A R1 de 02/11/2026 usa regras `governance-rules-v2-dev3`; veja [o relatório](../docs/R1_REVISAO_NOTAS.md).

A R3 usa regras `governance-rules-v2-dev4`, preparação `p3-v3` e pdfplumber 0.11.9. Atualize as dependências com `requirements.txt`. Como a ordem do texto pode mudar, reexecute a análise completa: `../docs/evidencias/R3/reanalisar.py <nova-pasta>`. O nome identifica uma pasta nova dentro das evidências R3; o comando não altera o banco nem os resultados antigos. Veja [o relatório da R3](../docs/R3_REVISAO_CONTEXTO_NOTAS.md).

## Atualizar modelo

Mantenha a pasta anterior, exporte/verifique o novo pacote com as mesmas categorias, ajuste MODEL_DIR e reinicie. Confira readiness e execute a validação antes de usar. Para voltar, restaure MODEL_DIR e reinicie. O formato de dados mantém `schema_version: 2`; versões mudam conforme artefatos, código e configuração. Alterações de categorias ou significado do resultado exigem rever o contrato.

## Estrutura e núcleo preservado

```text
src/privacy_service/main.py          HTTP, upload e concorrência
src/privacy_service/settings.py      Configuração e limites
src/privacy_service/extraction.py    Extração de PDF/TXT
src/privacy_service/segmentation.py  Trechos com posições e páginas
src/privacy_service/pipeline.py      Modelo, avaliação e manifesto
src/privacy_service/prepare.py       Preparação pelo terminal
vendor/privacy-policy-core/          Núcleo local instalado pela API
vendor/SNAPSHOT.json                 Origem e hashes da cópia local
config/                             Regras, recomendações e referências
model/                              Modelo exportado e integridade
tests/ e scripts/                   Testes e verificações reproduzíveis
```

O backend envia arquivo, idioma e escopo. O serviço retorna `schema_version`, `versions` e `result`, sem acessar PostgreSQL. Os dados de treinamento ficam no módulo científico; não são carregados para analisar uma política.

`pipeline.py` carrega os artefatos uma vez, confere sua integridade e aquece o modelo. `inference.py` classifica em lotes. `heuristics.py` procura evidências e conflitos; `scoring.py` resolve as perguntas e combina suas contribuições. `scoring_common.py` mantém validações e a escala. `recommendations.py` relaciona achados ao catálogo sem mudar a nota.

Esses arquivos, junto de extração, segmentação, regras, catálogo e `score_cli.py`, participam das identidades calculadas pelo código. Até comentários podem mudar os hashes expostos na análise ou no manifesto da CLI. Nesta simplificação eles foram preservados; esta seção documenta seu papel sem alterar as respostas.

O núcleo em `vendor/` é uma cópia necessária à publicação independente, não uma segunda implementação. Ao atualizar uma fonte autorizada, copie somente os arquivos alterados e atualize seus hashes em `SNAPSHOT.json`; confira a igualdade com a origem e rode os testes. Não regenere hashes do modelo para encobrir alterações nos artefatos.

`requirements.txt` instala o serviço e o núcleo local; `requirements.lock.txt` fixa dependências diretas e indiretas; `requirements-dev.txt` acrescenta testes e medição. Não remover um pacote somente por não existir um import direto no serviço.
