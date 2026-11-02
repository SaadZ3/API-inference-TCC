"""HTTP interno. Execute com um worker; o navegador usa a API Node."""
import asyncio
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.datastructures import UploadFile as StarletteUploadFile

from .extraction import IngestionError
from .settings import Settings

LOGGER = logging.getLogger(__name__)


def error(code, message, status):
    """Converte código, mensagem e status na resposta JSON de erro usada pelo backend."""
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


class BodyLimit:
    """Conta também corpos sem Content-Length antes de acumular o multipart."""
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await error("INVALID_INPUT", "Content-Length inválido.", 400)(scope, receive, send)
        if declared < 0:
            return await error("INVALID_INPUT", "Content-Length inválido.", 400)(scope, receive, send)
        if declared > self.max_bytes:
            return await error("FILE_TOO_LARGE", "Arquivo excede o limite permitido.", 413)(scope, receive, send)
        total = 0
        async def bounded_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    raise HTTPException(413)
            return message
        await self.app(scope, bounded_receive, send)


async def read_upload(request: Request, config: Settings):
    """Lê um multipart limitado e retorna conteúdo, nome e escopo, ou a resposta de erro."""
    async with asyncio.timeout(config.upload_timeout_seconds):
        async with request.form(max_files=1, max_fields=2, max_part_size=1024) as form:
            expected = ['document_scope', 'file', 'language']
            if sorted(key for key, _ in form.multi_items()) != expected:
                return error("INVALID_INPUT", "Campos de envio inválidos.", 400)
            upload = form["file"]
            if (not isinstance(upload, StarletteUploadFile) or not upload.filename
                    or form["language"] != "pt"
                    or form["document_scope"] not in ("full_document", "excerpt", "unknown")):
                return error("INVALID_INPUT", "Arquivo, idioma ou escopo inválido.", 400)
            content = await upload.read(config.limits.max_bytes + 1)
            if len(content) > config.limits.max_bytes:
                return error("FILE_TOO_LARGE", "Arquivo excede o limite permitido.", 413)
            filename, document_scope = upload.filename, form["document_scope"]
    return content, filename, document_scope


def create_app(settings=None, pipeline_factory=None):
    """Retorna a API usando a configuração e a fábrica de modelo recebidas, ou os padrões."""
    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = None
        app.state.busy = False
        app.state.task = None
        try:
            config = settings or Settings.from_env()
            app.state.settings = config
            factory = pipeline_factory
            if factory is None:
                from .pipeline import AnalysisPipeline
                factory = AnalysisPipeline
            app.state.pipeline = await asyncio.to_thread(factory, config)
        except Exception as exc:
            LOGGER.error("Inferência indisponível no início (%s). Confira pacote e configuração.", type(exc).__name__)
        yield
        if app.state.task is not None:
            # A thread nativa pode terminar depois de um timeout HTTP.
            await asyncio.gather(app.state.task, return_exceptions=True)

    app = FastAPI(title="Análise de políticas — TCC", version="0.4.0", lifespan=lifespan)
    # O limite total inclui até 64 KiB para campos e cabeçalhos multipart.
    # Ler .env aqui também permite refletir MAX_UPLOAD_BYTES no middleware.
    try:
        body_limit = (settings or Settings.from_env()).limits.max_bytes
    except (ValueError, OSError):
        body_limit = 10 * 1024 * 1024
    app.add_middleware(BodyLimit, max_bytes=body_limit + 64 * 1024)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        if exc.status_code == 413:
            return error("FILE_TOO_LARGE", "Arquivo excede o limite permitido.", 413)
        if exc.status_code in (400, 422):
            return error("INVALID_INPUT", "Envio multipart inválido.", 400)
        return error("INVALID_INPUT", "Rota ou método inválido.", exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error("INVALID_INPUT", "Envie file, language=pt e document_scope válido.", 400)

    @app.get("/health/live")
    # Informa que o processo responde, mesmo se o modelo não estiver disponível.
    async def live():
        return {"status": "ok", "phase": "P4"}

    @app.get("/health/ready")
    # Retorna disponibilidade, ocupação e versões do modelo já carregado.
    async def ready():
        if app.state.pipeline is None:
            return JSONResponse({"status": "not_ready", "reason": "Modelo ou regras indisponíveis."}, status_code=503)
        return {"status": "ready", "busy": app.state.busy, "versions": app.state.pipeline.versions}

    @app.get("/manifest")
    # Entrega a procedência da análise para o backend, sem caminhos locais.
    async def manifest():
        if app.state.pipeline is None:
            return error("INFERENCE_UNAVAILABLE", "Serviço de análise indisponível.", 503)
        return app.state.pipeline.manifest

    # O corpo é lido manualmente para limitar arquivos/campos ANTES do parsing.
    @app.post("/analyze", openapi_extra={
        "requestBody": {"required": True, "content": {"multipart/form-data": {
            "schema": {"type": "object", "required": ["file", "language", "document_scope"],
                       "additionalProperties": False, "properties": {
                "file": {"type": "string", "format": "binary"},
                "language": {"type": "string", "enum": ["pt"]},
                "document_scope": {"type": "string", "enum": ["full_document", "excerpt", "unknown"]},
            }}
        }}}
    })
    # Recebe PDF/TXT e devolve a análise; o bloqueio dura até terminar a tarefa nativa.
    async def analyze(request: Request):
        if app.state.pipeline is None or app.state.busy:
            return error("INFERENCE_UNAVAILABLE", "Serviço indisponível ou ocupado. Tente novamente.", 503)
        app.state.busy = True
        handed_off = False
        try:
            if request.headers.get("content-type", "").split(";")[0].lower() != "multipart/form-data":
                return error("INVALID_INPUT", "Envie um arquivo em multipart/form-data.", 400)
            config = app.state.settings
            upload = await read_upload(request, config)
            if isinstance(upload, JSONResponse):
                return upload
            content, filename, document_scope = upload
            deadline = time.monotonic() + config.analysis_timeout_seconds
            task = asyncio.create_task(asyncio.to_thread(
                app.state.pipeline.analyze, content, filename, document_scope, deadline))
            app.state.task = task
            handed_off = True
            def finished(completed):
                app.state.busy = False
                app.state.task = None
                if not completed.cancelled():
                    completed.exception()  # Consome falhas de trabalhos cujo HTTP já expirou.
            task.add_done_callback(finished)
            result = await asyncio.wait_for(asyncio.shield(task), config.analysis_timeout_seconds)
            return JSONResponse(result)
        except TimeoutError:
            return error("ANALYSIS_TIMEOUT", "Tempo limite excedido. Tente um documento menor.", 504)
        except IngestionError as exc:
            status = {"INVALID_INPUT": 400, "FILE_TOO_LARGE": 413, "UNSUPPORTED_FORMAT": 415}.get(exc.code, 422)
            return error(exc.code, exc.message, status)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.error("Falha de análise (%s).", type(exc).__name__)
            return error("INTERNAL_ERROR", "Não foi possível concluir a análise.", 500)
        finally:
            if not handed_off:
                app.state.busy = False

    return app


app = create_app()
