"""Configuração local do serviço; caminhos relativos à pasta tcc-inference."""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from .extraction import ExtractionLimits

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    scoring_rules: Path
    recommendations_catalog: Path = ROOT.parent / 'OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/config/recommendations.yaml'
    batch_size: int = 4
    cpu_threads: int = 4
    analysis_timeout_seconds: float = 120
    upload_timeout_seconds: float = 30
    limits: ExtractionLimits = field(default_factory=ExtractionLimits)

    def __post_init__(self):
        if not 1 <= self.batch_size <= 16 or not 1 <= self.cpu_threads <= 16:
            raise ValueError("BATCH_SIZE e CPU_THREADS devem estar entre 1 e 16.")
        for value in (self.analysis_timeout_seconds, self.upload_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Timeout deve ser positivo e finito.")

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        if os.getenv("DEVICE", "cpu") != "cpu":
            raise ValueError("Esta versão do serviço foi validada somente em CPU.")
        def path(name, default):
            return (ROOT / os.getenv(name, default)).resolve()
        return cls(
            model_dir=path("MODEL_DIR", "../OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/modelo"),
            scoring_rules=path("SCORING_RULES", "../OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/config/scoring.yaml"),
            recommendations_catalog=path('RECOMMENDATIONS_CATALOG', '../OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/config/recommendations.yaml'),
            batch_size=int(os.getenv("BATCH_SIZE", "4")),
            cpu_threads=int(os.getenv("CPU_THREADS", "4")),
            analysis_timeout_seconds=float(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "120")),
            upload_timeout_seconds=float(os.getenv("UPLOAD_TIMEOUT_SECONDS", "30")),
            limits=ExtractionLimits(max_bytes=int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))),
        )
