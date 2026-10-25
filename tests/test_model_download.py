"""Confere a integridade do download sem acessar a rede."""
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("start_railway", Path(__file__).resolve().parents[1] / "scripts/start_railway.py")
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class Download:
    def __init__(self, content):
        self.content = content

    def open(self, request, timeout):
        return io.BytesIO(self.content)


def expected_model(directory, content):
    (directory / "checksums.json").write_text(json.dumps({"model.safetensors": sha256(content).hexdigest()}))


def test_download_checks_integrity_and_reuses_existing_file(tmp_path):
    content = b"pesos de teste"
    expected_model(tmp_path, content)
    target = startup.ensure_weights(tmp_path, "https://example.com/model.safetensors", Download(content))
    assert target.read_bytes() == content
    assert not target.with_suffix(".part").exists()
    assert startup.ensure_weights(tmp_path, "") == target


def test_corrupt_download_is_not_installed(tmp_path):
    expected_model(tmp_path, b"modelo correto")
    with pytest.raises(ValueError, match="não corresponde"):
        startup.ensure_weights(tmp_path, "https://example.com/model.safetensors", Download(b"pagina de erro"))
    assert not (tmp_path / "model.safetensors").exists()
    assert not (tmp_path / "model.part").exists()


def test_missing_url_is_actionable(tmp_path):
    expected_model(tmp_path, b"modelo")
    with pytest.raises(ValueError, match="MODEL_WEIGHTS_URL"):
        startup.ensure_weights(tmp_path, "")


def test_corrupt_existing_weights_are_rejected(tmp_path):
    expected_model(tmp_path, b"modelo correto")
    (tmp_path / "model.safetensors").write_bytes(b"corrompido")
    with pytest.raises(ValueError, match="pesos locais"):
        startup.ensure_weights(tmp_path, "")


def test_redirect_cannot_downgrade_https():
    with pytest.raises(ValueError, match="HTTPS"):
        startup.HTTPSRedirects().redirect_request(None, None, 302, "", {}, "http://example.com/model")
