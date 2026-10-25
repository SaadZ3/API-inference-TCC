"""Exporta exclusivamente o checkpoint treinado e verifica a inferência portátil."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tratamento.cli import ROOT, save_json
from privacy_pipeline.inference import PrivacyClassifier


def write_model_card(run: dict, metrics: dict, thresholds: dict | None, output: Path) -> None:
    """Documenta resultados recebidos do Colab sem simular uma nova avaliação."""
    report = metrics["test"]
    labels = run["labels"]
    lines = ["---", "language:", f"- {run['language']}", "library_name: transformers", "tags:",
             "- privacy-policy", "- text-classification", "- opp-115", "---", "",
             "# Classificador de práticas de privacidade OPP-115", "",
             f"Modelo base: `{run['model']['name']}`, revisão `{run['model']['revision']}`.",
             f"Tarefa: **{run['task']}**. Seed: {run['seed']}. Unidade: trecho já segmentado.",
             "", "## Dados e preparação", "",
             "OPP-115, anotações originais unidas por segmento, dez categorias oficiais.",
             "Textos em português traduzidos pelo Google Planilhas; inglês preservado para comparação.",
             "Split próprio por política; políticas com trechos idênticos agrupadas no mesmo conjunto.",
             f"Revisão humana: `{run['human_review_status']}`. A aprovação cobre a amostra de 100 trechos e quatro avisos, não todo o corpus.",
             f"Hash do dataset: `{run['source_segments_sha256']}`.",
             f"Hash da tradução: `{run['translation_sha256']}`.", "",
             "| Split | Políticas | Trechos |", "|---|---:|---:|"]
    for name, split in run["splits"].items():
        lines.append(f"| {name} | {split['policies']} | {split['segments']} |")
    lines.extend(["", "## Treino e seleção", "",
                  f"Checkpoint: `{run['best_checkpoint']}`. Critério: {run['checkpoint_selection']}.",
                  f"Desbalanceamento: `{run['imbalance_method']}`; pesos calculados só no treino.",
                  "Os parâmetros completos, pesos, versões e procedência estão em `training_run.json`.",
                  "", "## Avaliação", "",
                  "Resultados do treino executado no Colab, importados dos relatórios do treinamento.",
                  "A verificação da exportação não refaz treinamento nem ajusta thresholds.",
                  f"Teste: {report['n_samples']} trechos; F1-macro **{report['f1_macro']:.6f}**; F1-micro **{report['f1_micro']:.6f}**; kappa **{report['cohen_kappa']}**.",
                  "Kappa: " + report["kappa_definition"] + ".", "",
                  "| Classe | Suporte | Precisão | Recall | F1 | Threshold |", "|---|---:|---:|---:|---:|---:|"])
    for i, label in enumerate(labels):
        row = report["per_class"][label]
        threshold = f"{thresholds['thresholds'][i]:.2f}" if thresholds else "argmax"
        lines.append(f"| {label} | {row['support']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {threshold} |")
    lines.extend(["", "## Inferência", "",
                  "```python", "from predict import PrivacyClassifier",
                  "model = PrivacyClassifier('caminho/do/pacote')",
                  "resultados = model.predict(['Não compartilhamos seus dados com terceiros.'])", "```", "",
                  f"Limite: {run['config']['training']['max_length']} tokens incluindo tokens especiais.",
                  "O início é preservado; `truncated=true` indica corte. Textos devem estar no idioma do modelo.",
                  "Sigmoid com limiar por classe no multi-label; softmax e argmax no multiclass.",
                  "A saída multi-label pode conter zero, um ou vários rótulos. Probabilidades não têm calibração garantida.",
                  "", "## Limitações e uso", "",
                  "Corpus pequeno, políticas antigas de sites predominantemente dos EUA, viés de domínio e tradução automática.",
                  "Classes raras, uma única seed e ausência de intervalos de confiança limitam comparações.",
                  "A classe identifica assunto/prática mencionada; não identifica por si só risco, conformidade legal ou polaridade da prática.",
                  "O score de política inteira pertence à avaliação da política e não está implementado neste pacote.",
                  "O modo multiclass usa somente trechos de rótulo único e não é diretamente comparável ao multi-label.",
                  "Respeitar as condições do corpus em readme.txt e a licença do modelo base. O pacote não concede uma nova licença aos dados.",
                  "Citar Wilson et al. (2016), The creation and analysis of a website privacy policy corpus, ACL.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def export_model(run_dir: Path, output: Path, verification_samples: int = 32, probability_atol: float = 0.005) -> dict:
    """Relê o pacote e compara suas saídas com previsões salvas da validação."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "completed" or run["algorithm"] != "transformer":
        raise ValueError("A exportação requer um treino Transformer concluído.")
    checkpoint = (run_dir / run["best_checkpoint"]).resolve()
    if not checkpoint.is_relative_to(run_dir.resolve()):
        raise ValueError("Checkpoint fora da pasta da execução.")
    if not (checkpoint / "config.json").is_file() or not any(checkpoint.glob("*.safetensors")):
        raise FileNotFoundError(f"Pesos treinados ausentes em {checkpoint}. Copie o melhor checkpoint do Colab/Drive; o ZIP de relatórios não contém os pesos.")
    if output.exists():
        raise FileExistsError(f"Destino já existe: {output}. Escolha um diretório novo.")
    if verification_samples < 1 or not 0 < probability_atol < 0.1:
        raise ValueError("Parâmetros de verificação inválidos.")
    labels = run["labels"]
    model, loading = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True, output_loading_info=True)
    if any(loading.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
        raise ValueError("O checkpoint não contém todos os pesos esperados.")
    if [model.config.id2label[i] for i in range(model.config.num_labels)] != labels:
        raise ValueError("Ordem de categorias diferente entre checkpoint e relatório.")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if tokenizer.truncation_side != "right":
        raise ValueError("Tokenizer não preserva o início dos textos como no treino.")
    thresholds = None
    if run["task"] == "multilabel":
        thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        if thresholds["labels"] != labels or thresholds["score_kind"] != "probability" or thresholds["fitted_on"] != "validation":
            raise ValueError("Thresholds não correspondem às probabilidades do modelo.")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    save_json(dict(enumerate(labels)), output / "label_mapping.json")
    save_json({"schema_version": 1, "task": run["task"], "language": run["language"],
               "max_length": run["config"]["training"]["max_length"],
               "long_text_strategy": run["config"]["training"]["long_text_strategy"]}, output / "inference.json")
    if thresholds:
        save_json(thresholds, output / "thresholds.json")
    save_json(run, output / "training_run.json")
    save_json(metrics, output / "metrics.json")
    shutil.copyfile(ROOT / "Treinamento/privacy_pipeline/inference.py", output / "predict.py")
    shutil.copyfile(ROOT / "Treinamento/requirements-inference.txt", output / "requirements.txt")
    write_model_card(run, metrics, thresholds, output)
    del model
    classifier = PrivacyClassifier(output)
    records = [json.loads(line) for line in (run_dir / "validation_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    # Ordem por ID, independente de acertos/erros ou resultados do teste.
    selected = sorted(records, key=lambda record: record["segment_id"])[:verification_samples]
    if not selected:
        raise ValueError("Previsões de validação ausentes para verificar a exportação.")
    actual = classifier.predict([record["text"] for record in selected])
    expected = np.array([[record["scores"][label] for label in labels] for record in selected])
    observed = np.array([[record["probabilities"][label] for label in labels] for record in actual])
    maximum = float(np.max(np.abs(expected - observed)))
    differences = [record["segment_id"] for record, predicted in zip(selected, actual) if record["predicted_labels"] != predicted["labels"]]
    passed = maximum <= probability_atol and not differences
    report = {"status": "verified" if passed else "failed", "checked_on": "validation", "sample_count": len(selected),
              "segment_ids": [record["segment_id"] for record in selected], "max_probability_difference": maximum,
              "probability_atol": probability_atol, "label_disagreements": differences,
              "device": "cpu", "torch_version": torch.__version__,
              "scope": "Carregamento offline, probabilidades e rótulos em amostra de validação; não é nova avaliação integral."}
    save_json(report, output / "export_verification.json")
    if not passed:
        raise ValueError(f"Exportação não validada: diferença máxima {maximum}, divergências {differences}. Consulte export_verification.json; não use o pacote como verificado.")
    save_json({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file()}, output / "checksums.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta o melhor checkpoint do treinamento")
    parser.add_argument("--config", type=Path, default=ROOT / "Treinamento/config/exportacao.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(export_model(args.run_dir or ROOT / config["run_dir"], args.output or ROOT / config["output_dir"],
                                  config["verification_samples"], config["probability_atol"]), ensure_ascii=False, indent=2))
