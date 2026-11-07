"""Uma instância por processo: extração, BERTimbau e regras existentes."""
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

from privacy_pipeline import inference, scoring, scoring_common, heuristics, recommendations
from . import extraction, segmentation, pdf_reading_order
from .settings import Settings

ARTIFACTS = (
    "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "label_mapping.json", "thresholds.json",
    "inference.json", "export_verification.json",
)


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Prazo de análise excedido.")


def evaluate(prepared, segments, rules, model_metadata, assessment_scope=None):
    """O mesmo núcleo de scoring usado na CLI de treinamento."""
    policy = {
        "schema_version": 1, "policy_id": prepared["source"]["sha256"],
        "language": "pt",
        # A qualidade da extração tem um bloqueio próprio, separado do escopo enviado.
        "complete": prepared["document_scope"] == "full_document",
        "applicability": {}, "model": model_metadata, "segments": segments,
    }
    if assessment_scope is not None:
        policy['assessment_scope'] = assessment_scope
    policy.update(document_scope=prepared['document_scope'], extraction=prepared['extraction'])
    return scoring.evaluate_policy(policy, rules)


class AnalysisPipeline:
    def __init__(self, settings: Settings):
        import torch
        self.settings = settings
        torch.set_num_threads(settings.cpu_threads)
        expected = json.loads((settings.model_dir / "checksums.json").read_text(encoding="utf-8"))
        hashes = {name: file_hash(settings.model_dir / name) for name in ARTIFACTS}
        if any(expected.get(name) != digest for name, digest in hashes.items()):
            raise ValueError("Integridade do pacote exportado não confere.")
        self.rules = heuristics.load_rules(settings.scoring_rules)
        self.recommendations = recommendations.load_catalog(settings.recommendations_catalog)
        self.classifier = inference.PrivacyClassifier(settings.model_dir, device="cpu")
        self.segmenter = segmentation.PolicySegmenter(settings.model_dir)
        if (self.classifier.task != "multilabel"
                or self.classifier.settings.get("language") != "pt"
                or tuple(self.classifier.labels) != scoring_common.LABELS):
            raise ValueError("Modelo incompatível com o contrato de análise.")
        self.model_metadata = {
            "files_sha256": {k: v for k, v in hashes.items() if k != "export_verification.json"},
            "inference": self.classifier.settings, "device": "cpu",
            "batch_size": settings.batch_size,
            "thresholds": json.loads((settings.model_dir / "thresholds.json").read_text(encoding="utf-8")),
        }
        runtime = {name: version(name) for name in ("torch", "transformers", "tokenizers", "pypdf", "pdfplumber", "pdfminer.six")}
        model_identity = {"files_sha256": hashes, "runtime": runtime,
                          "inference_code_sha256": file_hash(inference.__file__),
                          "batch_size": settings.batch_size, "cpu_threads": settings.cpu_threads}
        preprocessing_identity = {
            "version": segmentation.PREPROCESSING_VERSION,
            "tokenizer_sha256": self.segmenter.tokenizer_sha256,
            "files_sha256": {name: file_hash(module.__file__) for name, module in
                            (("extraction.py", extraction), ("segmentation.py", segmentation), ("pdf_reading_order.py", pdf_reading_order))},
            "limits": asdict(settings.limits), "pypdf": runtime["pypdf"],
            "pdfplumber": runtime["pdfplumber"], "pdfminer.six": runtime["pdfminer.six"],
            "tokenizers": runtime["tokenizers"],
        }
        scoring_identity = {
            'rules_version': self.rules.data['version'],
            'rules_sha256': scoring_common.content_hash(self.rules.data),
            'rules_file_sha256': file_hash(settings.scoring_rules),
            'rules': self.rules.data,
            'code_sha256': {**scoring.code_hashes(), 'pipeline.py': file_hash(__file__)},
            'pyyaml': version('PyYAML'),
        }
        self.versions = {
            'model': 'bertimbau:' + scoring_common.content_hash(model_identity),
            'preprocessing': segmentation.PREPROCESSING_VERSION + ':' + scoring_common.content_hash(preprocessing_identity),
            'scoring': self.rules.data['version'] + ':' + scoring_common.content_hash(scoring_identity),
            'recommendations': self.recommendations.identity,
        }
        self.manifest = {
            'schema_version': 2, 'versions': self.versions, 'model': model_identity,
            'preprocessing': preprocessing_identity, 'scoring': scoring_identity,
            'recommendations': self.recommendations.manifest,
            'service_code_sha256': file_hash(__file__), 'runtime': runtime,
            'device': 'cpu', 'batch_size': settings.batch_size, 'cpu_threads': settings.cpu_threads,
            'analysis_timeout_seconds': settings.analysis_timeout_seconds,
        }
        # Readiness exige também uma passagem real pelo modelo.
        self.classifier.predict(["Coletamos dados pessoais para prestar nossos serviços."], batch_size=1)

    def analyze(self, content, filename, document_scope, deadline, assessment_scope=None):
        check_deadline(deadline)
        document = extraction.extract_document(
            content, filename, document_scope=document_scope, limits=self.settings.limits)
        check_deadline(deadline)
        prepared = self.segmenter.prepare(document)
        check_deadline(deadline)
        segments = []
        for start in range(0, len(prepared["segments"]), self.settings.batch_size):
            batch = prepared["segments"][start:start + self.settings.batch_size]
            predictions = self.classifier.predict(
                [segment["text"] for segment in batch], batch_size=self.settings.batch_size)
            for segment, prediction in zip(batch, predictions, strict=True):
                if prediction["text"] != segment["text"] or prediction["n_tokens"] != segment["n_tokens"]:
                    raise ValueError("Tokenização inconsistente entre preparação e inferência.")
                segments.append({**segment, **prediction})
            check_deadline(deadline)
        evaluation = evaluate(prepared, segments, self.rules, self.model_metadata, assessment_scope)
        evaluation = recommendations.apply_recommendations(evaluation, self.recommendations)
        check_deadline(deadline)
        return {
            "schema_version": evaluation["schema_version"],
            "versions": self.versions,
            "result": {
                "document_scope": document_scope, "source": prepared["source"],
                "preprocessing": prepared["preprocessing"], "extraction": prepared["extraction"],
                "segments": segments, "evaluation": evaluation,
            },
        }
