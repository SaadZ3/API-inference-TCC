"""Classifica trechos fornecidos ou reavalia previsões sem carregar o modelo."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from . import scoring, recommendations
from .heuristics import load_rules
from .scoring_common import validate_policy

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_policy(policy: dict, model_dir: Path, device: str, batch_size: int) -> dict:
    """Adaptador: apenas este caminho importa a stack de inferência."""
    from .inference import PrivacyClassifier

    validate_policy(policy, predictions=False)
    classifier = PrivacyClassifier(model_dir, device=device)
    if classifier.task != "multilabel" or classifier.settings.get("language") != "pt":
        raise ValueError("A avaliação exige o modelo multi-label em português.")
    predictions = classifier.predict([s["text"] for s in policy["segments"]], batch_size=batch_size)
    filenames = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.txt", "label_mapping.json", "thresholds.json", "inference.json"]
    model = {"files_sha256": {name: _file_hash(model_dir / name) for name in filenames if (model_dir / name).is_file()},
             "inference": classifier.settings, "device": device, "batch_size": batch_size,
             "thresholds": json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))}
    return {**policy, "model": model,
            "segments": [{**s, **p} for s, p in zip(policy["segments"], predictions, strict=True)]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação da qualidade da informação documentada.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON de uma política já segmentada.")
    source.add_argument("--predictions", type=Path, help="JSON de previsões anteriores; não carrega PyTorch.")
    parser.add_argument("--rules", type=Path, help="Configuração das regras de avaliação.")
    parser.add_argument("--no-fallback", action="store_true", help="Experimento: buscar somente nas categorias previstas.")
    parser.add_argument('--recommendations', type=Path, help='Catálogo de recomendações.')
    parser.add_argument("--model-dir", type=Path, default=ROOT / "Treinamento/modelo")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.rules is None:
        args.rules = ROOT / 'Treinamento/config/scoring.yaml'
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.output_dir.exists():
        raise FileExistsError(f"Use uma pasta de saída nova: {args.output_dir}")
    source_path = args.input or args.predictions
    policy = json.loads(source_path.read_text(encoding="utf-8"))
    rules = load_rules(args.rules)
    if args.predictions and isinstance(policy, dict) and 'result' in policy:
        policy = scoring.from_service_response(policy)
    validate_policy(policy, predictions=args.predictions is not None)
    if args.input:
        LOGGER.info("Classificando %d trechos em %s.", len(policy["segments"]), args.device)
        predictions = classify_policy(policy, args.model_dir, args.device, args.batch_size)
    else:
        LOGGER.info("Reutilizando previsões; não é necessário retreinar ou carregar o modelo.")
        predictions = policy
    report = scoring.evaluate_policy(predictions, rules, fallback=False if args.no_fallback else None)
    if args.recommendations:
        catalog = recommendations.load_catalog(args.recommendations)
        report = recommendations.apply_recommendations(report, catalog)
    args.output_dir.mkdir(parents=True)
    for name, value in {"input.json": policy, "predictions.json": predictions, "evaluation.json": report,
                        "scoring_rules.json": rules.data}.items():
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    import platform
    import importlib.metadata
    manifest = {'method': 'documentary_information_quality', 'source_file_sha256': _file_hash(source_path),
                'rules_file_sha256': _file_hash(args.rules), 'rules_sha256': report['rules_sha256'],
                'code_sha256': {**scoring.code_hashes(), 'score_cli.py': _file_hash(Path(__file__))},
                'predictions_sha256': report['predictions_sha256'], 'profile_sha256': report['profile_sha256'],
                'fallback_enabled': report['fallback_enabled'], 'python': platform.python_version(),
                'pyyaml': importlib.metadata.version('PyYAML')}
    if args.recommendations:
        manifest['recommendations'] = {'identity': catalog.identity, **catalog.manifest}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / "REPORT.md").write_text(scoring.render_report(report), encoding="utf-8")
    if args.recommendations:
        with (args.output_dir / 'REPORT.md').open('a', encoding='utf-8') as stream:
            stream.write(recommendations.render_recommendations(report))
    LOGGER.info("Avaliação salva em %s. Estado: %s; nota: %s.", args.output_dir,
                report['score_status'], report["score"])


if __name__ == "__main__":
    main()
