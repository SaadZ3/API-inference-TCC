"""Baixa os pesos da Release e confere SHA-256 antes de iniciar o serviço."""
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("O download exige HTTPS também nos redirecionamentos.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(model_dir, url, opener=None):
    expected = json.loads((model_dir / "checksums.json").read_text(encoding="utf-8"))["model.safetensors"]
    target = model_dir / "model.safetensors"
    if target.exists():
        if file_hash(target) != expected:
            raise ValueError("Os pesos locais não correspondem ao checksums.json.")
        return target
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Configure MODEL_WEIGHTS_URL com o endereço HTTPS público do anexo da Release.")
    temporary = target.with_suffix(".part")
    try:
        print("Baixando pesos do modelo...", flush=True)
        request = Request(url, headers={"User-Agent": "tcc-inference", "Accept": "application/octet-stream"})
        with (opener or build_opener(HTTPSRedirects())).open(request, timeout=60) as response:
            with temporary.open("wb") as output:
                total = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    total += len(block)
                    if total > 1024 * 1024 * 1024:
                        raise ValueError("O download ultrapassou o limite de 1 GiB.")
                    output.write(block)
        if file_hash(temporary) != expected:
            raise ValueError("O arquivo baixado não corresponde ao modelo esperado. Confira a Release.")
        temporary.replace(target)
        print("Modelo baixado e integridade confirmada.", flush=True)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def main():
    try:
        model_dir = (ROOT / os.getenv("MODEL_DIR", "model")).resolve()
        ensure_weights(model_dir, os.getenv("MODEL_WEIGHTS_URL", ""))
    except ValueError as exc:
        print(f"Não foi possível preparar o modelo: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # URLs assinadas podem conter segredos; não registrar o erro HTTP completo.
        print(f"Não foi possível baixar o modelo ({type(exc).__name__}). Confira a URL da Release e a conexão.", file=sys.stderr)
        return 1
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    # A mesma porta atende o healthcheck IPv4 e a rede privada IPv6 do Railway.
    with socket.create_server(("::", port), family=socket.AF_INET6, dualstack_ipv6=True) as listener:
        config = uvicorn.Config("privacy_service.main:app", host="::", port=port, workers=1)
        uvicorn.Server(config).run(sockets=[listener])
    return 0


if __name__ == "__main__":
    sys.exit(main())
