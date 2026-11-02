# Compartilha tokenizer e configuração local entre os testes, sem baixar modelos.
from pathlib import Path
import os
import pytest

from privacy_service.segmentation import PolicySegmenter

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@pytest.fixture(scope="session")
def model_dir():
    return Path(os.environ.get("TCC_TEST_MODEL_DIR", str(
        Path(__file__).resolve().parents[1] / "model"
    )))


@pytest.fixture(scope="session")
def segmenter(model_dir):
    # Usa o tokenizer real exportado, sem baixar nada e sem carregar os pesos.
    return PolicySegmenter(model_dir)
