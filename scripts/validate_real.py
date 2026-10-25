"""Validação opt-in com pesos reais; nenhuma chamada ao banco ou ao treinamento."""
import argparse
import json
import platform
from pathlib import Path
import threading
import time

import psutil
from fastapi.testclient import TestClient

from privacy_pipeline.scoring import evaluate_policy
from privacy_service.main import create_app
from privacy_service.pipeline import AnalysisPipeline
from privacy_service.settings import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = Settings(args.model_dir, args.rules)
    measurements = {"platform": platform.platform(), "processor": platform.processor(),
                    "logical_cpus": psutil.cpu_count(), "device": "cpu",
                    "batch_size": settings.batch_size, "cpu_threads": settings.cpu_threads}
    process = psutil.Process()
    memory = {"rss_peak_bytes": 0, "private_peak_bytes": 0}
    stop = threading.Event()
    def sample():
        while not stop.wait(0.02):
            info = process.memory_info()
            memory["rss_peak_bytes"] = max(memory["rss_peak_bytes"], info.rss)
            memory["private_peak_bytes"] = max(memory["private_peak_bytes"], getattr(info, "private", 0))
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    loaded = []
    def factory(config):
        start = time.perf_counter()
        pipeline = AnalysisPipeline(config)
        measurements["load_and_warmup_seconds"] = time.perf_counter() - start
        loaded.append(pipeline)
        return pipeline
    app = create_app(settings, factory)
    records = []
    try:
        with TestClient(app) as client:
            assert client.get("/health/ready").status_code == 200
            manifest = client.get("/manifest").json()
            (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            cases = [
                ("synthetic", "policy.txt", "Coletamos seu nome e email para fornecer o serviço.\n\nVocê pode solicitar a exclusão dos dados.".encode(), "full_document"),
                ("public_pdf", args.pdf.name, args.pdf.read_bytes(), "full_document"),
                ("public_pdf_excerpt", args.pdf.name, args.pdf.read_bytes(), "excerpt"),
                ("limit_100k", "limit.txt", ("Coletamos dados para prestar serviços. " * 3000)[:100000].encode(), "full_document"),
            ]
            for name, filename, content, scope in cases:
                start = time.perf_counter()
                response = client.post("/analyze", files={"file": (filename, content)},
                                       data={"language": "pt", "document_scope": scope})
                elapsed = time.perf_counter() - start
                assert response.status_code == 200, response.text
                body = response.json()
                result = body["result"]
                assert all(not item["truncated"] for item in result["segments"])
                for item in result["segments"]:
                    assert result["extraction"]["text"][item["char_start"]:item["char_end"]] == item["text"]
                if scope == "excerpt":
                    assert result["evaluation"]["score"] is None
                (args.output_dir / (name + ".json")).write_text(
                    json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
                records.append({"case": name, "seconds": elapsed, "bytes": len(content),
                                "segments": len(result["segments"]),
                                "pages": result["extraction"]["page_count"],
                                "status": result["evaluation"]["input_status"],
                                "score": result["evaluation"]["score"]})
            assert len(loaded) == 1
            # Mesmo modelo e textos da rodada anterior, comparando com previsões salvas.
            previous = Path(__file__).resolve().parents[2] / "OPP-115 Treinamento/Tratamento e Treinamento/Treinamento/tests/fixtures/politica_real_predictions.json"
            old = json.loads(previous.read_text(encoding="utf-8"))
            pipeline = loaded[0]
            predictions = pipeline.classifier.predict([s["text"] for s in old["segments"]], batch_size=4)
            largest_delta = 0.0
            for old_segment, prediction in zip(old["segments"], predictions, strict=True):
                assert old_segment["labels"] == prediction["labels"]
                assert old_segment["n_tokens"] == prediction["n_tokens"]
                for label, probability in prediction["probabilities"].items():
                    largest_delta = max(largest_delta, abs(probability - old_segment["probabilities"][label]))
            assert largest_delta < 0.0001, largest_delta
            updated = {**old, "model": pipeline.model_metadata,
                       "segments": [{"segment_id": s["segment_id"], **p}
                                    for s, p in zip(old["segments"], predictions, strict=True)]}
            before, after = evaluate_policy(old, pipeline.rules), evaluate_policy(updated, pipeline.rules)
            for key in ("score", "input_status", "dimensions", "findings", "limitations"):
                assert before[key] == after[key], key
            measurements["equivalence"] = {"segments": len(predictions), "labels_equal": True,
                                           "evaluation_equal": True, "max_probability_delta": largest_delta}
            measurements["model_loads"] = len(loaded)
    finally:
        stop.set()
        monitor.join()
    measurements.update(memory)
    measurements["cases"] = records
    (args.output_dir / "measurements.json").write_text(
        json.dumps(measurements, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(measurements, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
