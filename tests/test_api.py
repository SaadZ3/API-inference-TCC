from dataclasses import replace
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
import pytest

from privacy_service.main import create_app
from privacy_service.settings import Settings
from privacy_service.extraction import ExtractionLimits, IngestionError

CONFIG = Settings(Path("unused"), Path("unused"))


class FakePipeline:
    versions = {"model": "test", "preprocessing": "test", "scoring": "test"}
    manifest = {"versions": versions}

    def analyze(self, content, filename, scope, deadline):
        return {"content_length": len(content), "filename": filename, "scope": scope}


def send(client, **kwargs):
    args = {"files": {"file": ("policy.txt", b"Coletamos dados.")},
            "data": {"language": "pt", "document_scope": "full_document"}}
    args.update(kwargs)
    return client.post("/analyze", **args)


def test_load_once_and_manifest():
    calls = []
    def factory(config):
        calls.append(config)
        return FakePipeline()
    with TestClient(create_app(CONFIG, factory)) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.get("/health/live").json()["phase"] == "P4"
        assert client.get("/manifest").json()["versions"] == FakePipeline.versions
        assert send(client).status_code == 200
        assert send(client).status_code == 200
        assert len(calls) == 1


def test_load_failure():
    def fail(config):
        raise ValueError("private path must not leak")
    with TestClient(create_app(CONFIG, fail)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert send(client).json()["error"]["code"] == "INFERENCE_UNAVAILABLE"
        assert client.get("/manifest").status_code == 503


def test_retired_routes_and_method_selection_are_unavailable():
    with TestClient(create_app(CONFIG, lambda _: FakePipeline())) as client:
        for route in ['/v1/analyze', '/v2/analyze']:
            assert client.post(route).status_code == 404
        for field in ['profile_id', 'method', 'version']:
            assert send(client, data={'language': 'pt', 'document_scope': 'full_document', field: 'custom'}).status_code == 400


@pytest.mark.parametrize("data", [
    {}, {"language": "en", "document_scope": "full_document"},
    {"language": "pt", "document_scope": "invalid"},
    {"language": "pt", "document_scope": "unknown", "url": "https://example.com"},
    {"language": "pt", "document_scope": ["excerpt", "full_document"]},
    {"language": "pt", "document_scope": "x" * 1025},
])
def test_invalid_fields(data):
    with TestClient(create_app(CONFIG, lambda _: FakePipeline())) as client:
        assert send(client, data=data).status_code == 400
        assert send(client).status_code == 200


def test_invalid_body_and_multiple_files():
    with TestClient(create_app(CONFIG, lambda _: FakePipeline())) as client:
        assert client.post("/analyze", json={}).status_code == 400
        assert send(client, files=[("file", ("a.txt", b"a")), ("file", ("b.txt", b"b"))]).status_code == 400
        response = client.post("/analyze", content=b"bad", headers={"content-type": "multipart/form-data"})
        assert response.status_code == 400


def test_size_limit_and_chunked_body():
    config = replace(CONFIG, limits=ExtractionLimits(max_bytes=10))
    with TestClient(create_app(config, lambda _: FakePipeline())) as client:
        assert send(client).status_code == 413
        response = client.post("/analyze", content=b"x" * 70000,
                               headers={"content-type": "multipart/form-data; boundary=a"})
        assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
        # No Content-Length: exceed limit while streaming a valid multipart prefix.
        def body():
            yield b'--a\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\n'
            yield b"x" * 70000
            yield b"\r\n--a--\r\n"
        response = client.post("/analyze", content=body(),
                               headers={"content-type": "multipart/form-data; boundary=a"})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


@pytest.mark.parametrize("code,status", [
    ("UNSUPPORTED_FORMAT", 415), ("OCR_REQUIRED", 422), ("UNREADABLE_FILE", 422),
    ("FILE_TOO_LARGE", 413), ("INVALID_INPUT", 400),
])
def test_ingestion_errors(code, status):
    class Failure(FakePipeline):
        def analyze(self, *args):
            raise IngestionError(code, "Arquivo inválido.")
    with TestClient(create_app(CONFIG, lambda _: Failure())) as client:
        response = send(client)
        assert response.status_code == status
        assert response.json()["error"]["code"] == code


def test_unexpected_error_does_not_leak():
    class Failure(FakePipeline):
        def analyze(self, *args):
            raise RuntimeError("document text and secret path")
    with TestClient(create_app(CONFIG, lambda _: Failure())) as client:
        response = send(client)
        assert response.status_code == 500
        assert "secret" not in response.text


def test_timeout_keeps_slot_until_thread_finishes():
    finished = threading.Event()
    release = threading.Event()
    class Slow(FakePipeline):
        def analyze(self, *args):
            release.wait(2)
            finished.set()
            return {}
    config = replace(CONFIG, analysis_timeout_seconds=0.02)
    with TestClient(create_app(config, lambda _: Slow())) as client:
        try:
            assert send(client).json()["error"]["code"] == "ANALYSIS_TIMEOUT"
            assert client.get("/health/ready").json()["busy"] is True
            assert send(client).status_code == 503
        finally:
            release.set()
            assert finished.wait(2)
        for _ in range(100):
            if not client.get("/health/ready").json()["busy"]:
                break
            time.sleep(0.01)
        assert client.get("/health/ready").json()["busy"] is False
        assert send(client).status_code == 200
