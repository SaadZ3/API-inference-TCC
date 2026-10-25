# Publicar a API Python pelo site do Railway

Este repositório contém o serviço, uma cópia do núcleo Python em `vendor/`, as regras em `config/` e os metadados do modelo em `model/`. Não precisa da pasta irmã `OPP-115 Treinamento` para executar. O treinamento continua no projeto original.

Validação local em 25/10/2026: 81 testes passaram; a imagem Linux foi construída com `pip check` sem conflitos. Dentro do contêiner, `/health/ready` respondeu 200 em IPv4 e IPv6 e uma análise HTTP com os pesos reais respondeu 200. Os pesos foram montados localmente nesse teste; o download da Release real só poderá ser conferido depois de publicar o anexo. O verificador de download foi testado com arquivos válidos e corrompidos, sem rede.

## 1. Enviar o código pelo GitHub

Envie as alterações deste repositório pelo controle de versão do VS Code. Confira se `Dockerfile`, `railway.json`, `scripts/start_railway.py`, `vendor/`, `config/` e os arquivos pequenos de `model/` aparecem no GitHub.

O `.gitignore` exclui ambientes, segredos, resultados e `model/model.safetensors`. Não force a inclusão dos pesos no commit. Eles têm aproximadamente 436 MB e serão enviados como anexo de uma Release.

## 2. Publicar os pesos em uma Release

No repositório **público** do GitHub:

1. Abra **Releases → Create a new release** (ou **Draft a new release**).
2. Crie a tag `model-v1`, selecione a branch com o código e use o título `Modelo de inferência v1`.
3. Na área de anexos, arraste `model/model.safetensors` do seu computador.
4. Aguarde o envio terminar e clique em **Publish release**.
5. Em **Assets**, copie o endereço do anexo `model.safetensors`.

O endereço deve ter este formato:

```text
https://github.com/USUARIO/REPOSITORIO/releases/download/model-v1/model.safetensors
```

Use o endereço do arquivo, não a página da Release nem os arquivos automáticos “Source code”. O anexo será público, como o repositório. A API verifica o SHA-256 registrado em `model/checksums.json`; um arquivo incorreto impede a inicialização.

## 3. Criar o serviço no Railway

1. Abra o projeto que contém o backend.
2. Clique em **New → GitHub Repo** e escolha este repositório.
3. Nomeie o serviço **tcc-inference**.
4. Deixe **Root Directory** no padrão: o `Dockerfile` está na raiz do repositório.
5. Remova comandos manuais antigos de build/start, caso existam. O Dockerfile define ambos.
6. Na aba **Variables**, adicione:

| Variável | Valor |
|---|---|
| `MODEL_WEIGHTS_URL` | URL pública do anexo da Release |
| `PORT` | `8000` |

Os caminhos `/app/model` e `/app/config`, CPU, um processo e os limites usuais já estão configurados na imagem. Não copie o `.env` do computador com caminhos Windows. Não configure `DATABASE_URL` neste serviço.

Mantenha uma réplica, desative a suspensão automática e disponibilize aproximadamente 3 GiB de RAM como margem inicial. Esse serviço requer recursos pagos suficientes; a memória efetiva deve ser medida depois da publicação.

O `railway.json` define `/health/ready` como verificação de saúde e permite até 600 segundos para baixar e carregar o modelo. Não gere domínio público. Não precisa de volume nesta configuração: os pesos são baixados novamente quando um novo contêiner é criado. Não remova o anexo da Release enquanto esta versão estiver em uso.

## 4. Publicar e conferir

Aplique as alterações e acompanhe os logs. A primeira execução deve indicar:

```text
Baixando pesos do modelo...
Modelo baixado e integridade confirmada.
```

Depois, o Uvicorn inicia e carrega o modelo. O deploy só fica saudável quando `/health/ready` retorna 200. Um processo iniciado com readiness 503 ainda não está pronto para analisar.

## 5. Conectar o backend

No serviço Node, configure:

```env
INFERENCE_URL=http://tcc-inference.railway.internal:8000
INFERENCE_TIMEOUT_MS=160000
```

Confira o nome real em **Settings → Networking → Private Networking** do Python. Se o domínio interno for diferente, use o valor mostrado no painel. Os dois serviços precisam estar no mesmo projeto e ambiente. Aplique as variáveis do backend e aguarde sua atualização.

Faça uma análise de um TXT curto pela aplicação para conferir a integração completa. `/api/health` do backend confirma apenas o Node, não a comunicação com o modelo.

## Diagnóstico

- **Problema:** erro HTTP ao baixar os pesos. **Causa:** Release em rascunho, endereço da página ou arquivo indisponível. **Correção:** publique a Release e use o link direto de Assets.
- **Problema:** integridade do modelo inválida. **Causa:** anexo diferente do arquivo local esperado. **Correção:** envie o `model.safetensors` que acompanha estes metadados.
- **Problema:** processo encerrado por falta de memória. **Causa:** limite insuficiente para PyTorch e o modelo. **Correção:** ajuste os recursos e mantenha um processo.
- **Problema:** build não encontra uma versão Python fixada. **Causa:** as dependências foram validadas originalmente no Windows. **Correção:** copie o erro de instalação para conferirmos compatibilidade Linux; não substitua versões arbitrariamente.

## Atualizar o núcleo ou modelo

`vendor/` é uma cópia do código do projeto de treinamento, não uma instalação de um pacote público homônimo. `vendor/SNAPSHOT.json` registra a origem e os hashes dos arquivos copiados. Atualizações no código original não entram automaticamente neste repositório: copie a nova versão, atualize o registro e valide novamente.

Para outro modelo, atualize todos os metadados e hashes juntos e publique os novos pesos em outra tag. Não substitua silenciosamente um anexo já utilizado.

Referências: [Releases no GitHub](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), [Dockerfiles no Railway](https://docs.railway.com/builds/dockerfiles), [rede privada](https://docs.railway.com/networking/private-networking).
