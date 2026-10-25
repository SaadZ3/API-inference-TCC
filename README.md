# Serviço Python de análise

Para publicar pelo site do Railway, siga [RAILWAY.md](RAILWAY.md). O repositório agora inclui o núcleo em `vendor/`, regras em `config/` e metadados em `model/`. Os pesos grandes são distribuídos como anexo de Release.

O projeto usa um único método de avaliação da qualidade da informação documentada, com notas de **0 a 3**. Consulte [a atualização do método](../docs/AVALIACAO_UNIFICADA.md).

P4 disponível: PDF digital/TXT UTF-8 → extração → BERTimbau → score com evidências. Serviço interno, independente do banco. O navegador será integrado pelo Node nas P5/P6.

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

## Atualizar modelo

Mantenha a pasta anterior, exporte/verifique o novo pacote com as mesmas categorias, ajuste MODEL_DIR e reinicie. Confira readiness e execute a validação antes de usar. Para voltar, restaure MODEL_DIR e reinicie. O formato de dados mantém `schema_version: 2`; versões mudam conforme artefatos, código e configuração. Alterações de categorias ou significado do resultado exigem rever o contrato.
