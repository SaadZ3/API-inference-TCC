"""Fine-tuning com Trainer, perda ponderada e seleção apenas na validação."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset
from scipy.special import expit, softmax
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from tratamento.cli import ROOT, save_json
from privacy_pipeline.experiments import create_run, evaluate_scores, load_experiment_data, targets
from privacy_pipeline.metrics import classification_metrics, imbalance_weights


class WeightedTrainer(Trainer):
    """Calcula BCE com pos_weight ou entropia cruzada com pesos do treino."""

    def __init__(self, *args: Any, task: str, class_weights: np.ndarray, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task = task
        self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        # A perda usa média do minibatch; o Trainer deve escalá-la na acumulação.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model: torch.nn.Module, inputs: dict, return_outputs: bool = False,
                     num_items_in_batch: torch.Tensor | None = None) -> Any:
        """Converte logits para float32 para estabilidade da perda em fp16."""
        values = dict(inputs)
        labels = values.pop("labels")
        outputs = model(**values)
        logits = outputs.logits.float()
        weights = self.class_weights.to(logits.device)
        if self.task == "multilabel":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=weights)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels.long(), weight=weights)
        return (loss, outputs) if return_outputs else loss


def build_dataset(frame: Any, tokenizer: Any, labels: list[str], task: str,
                  language: str, max_length: int) -> tuple[Dataset, dict]:
    """Audita cada truncamento; os textos integrais permanecem nos relatórios."""
    tokenized = tokenizer(frame[f"texto_{language}"].tolist(), truncation=False, padding=False, verbose=False)
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    audit = {"segments": len(frame), "truncated_segments": sum(n > max_length for n in lengths),
             "max_tokens": max(lengths), "max_length": max_length,
             "strategy": "head_truncation_with_audit",
             "truncated_segment_ids": frame.loc[np.array(lengths) > max_length, "segment_id"].tolist()}
    encoded = tokenizer(frame[f"texto_{language}"].tolist(), truncation=True, max_length=max_length, padding=False)
    y = targets(frame, labels, task)
    encoded["labels"] = y.astype(np.float32).tolist() if task == "multilabel" else y.tolist()
    return Dataset.from_dict(dict(encoded)), audit


def trainer_metric_function(labels: list[str], task: str) -> Any:
    """Seleciona checkpoint com F1-macro em limiar fixo 0,5, sem usar teste."""
    def compute(evaluation: Any) -> dict[str, float]:
        logits = evaluation.predictions[0] if isinstance(evaluation.predictions, tuple) else evaluation.predictions
        predicted = (expit(logits) >= 0.5).astype(int) if task == "multilabel" else logits.argmax(axis=1)
        metrics = classification_metrics(evaluation.label_ids, predicted, labels, task)
        return {name: metrics[name] for name in ["f1_macro", "f1_micro"]}
    return compute


def run_training(config: dict, language: str, task: str, run_name: str, allow_cpu: bool = False) -> dict:
    """Treina até quatro épocas por padrão e avalia o melhor checkpoint uma vez."""
    start = time.perf_counter()
    cuda = torch.cuda.is_available()
    if not cuda and not allow_cpu:
        raise ValueError("CUDA não disponível. Use GPU no Colab ou --allow-cpu para execução explícita em CPU.")
    settings = config["training"]
    if settings["long_text_strategy"] != "head_truncation_with_audit":
        raise ValueError("Estratégia para trechos longos não implementada.")
    if not 2e-5 <= settings["learning_rate"] <= 5e-5 or not 3 <= settings["epochs"] <= 5:
        raise ValueError("Use learning_rate entre 2e-5 e 5e-5 e entre 3 e 5 épocas.")
    set_seed(int(config["seed"]))
    splits, labels, metadata = load_experiment_data(config, task)
    weights = imbalance_weights(targets(splits["train"], labels, task), task, len(labels))
    model_spec = config["models"][language]
    cache_dir = ROOT / ".cache/huggingface"
    tokenizer = AutoTokenizer.from_pretrained(model_spec["name"], revision=model_spec["revision"], cache_dir=cache_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_spec["name"], revision=model_spec["revision"], cache_dir=cache_dir,
        num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)},
        problem_type="multi_label_classification" if task == "multilabel" else "single_label_classification")
    if not 2 < settings["max_length"] <= model.config.max_position_embeddings:
        raise ValueError("max_length ultrapassa o limite do modelo ou é inválido.")
    output = create_run(config, run_name)
    data, audit = {}, {}
    for name, frame in splits.items():
        data[name], audit[name] = build_dataset(frame, tokenizer, labels, task, language, settings["max_length"])
    save_json(audit, output / "truncation_audit.json")
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), learning_rate=settings["learning_rate"],
        num_train_epochs=settings["epochs"], warmup_ratio=settings["warmup_ratio"],
        weight_decay=settings["weight_decay"], per_device_train_batch_size=settings["train_batch_size"],
        per_device_eval_batch_size=settings["eval_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=25,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        save_total_limit=settings["save_total_limit"], fp16=bool(cuda and settings["fp16_on_cuda"]),
        save_only_model=True,
        gradient_checkpointing=settings["gradient_checkpointing"],
        seed=config["seed"], data_seed=config["seed"], report_to=[],
        dataloader_num_workers=0, dataloader_pin_memory=cuda, optim="adamw_torch",
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=data["train"], eval_dataset=data["validation"],
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if cuda else None),
        compute_metrics=trainer_metric_function(labels, task), task=task, class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=settings["early_stopping_patience"])],
    )
    metadata.update({"algorithm": "transformer", "model": model_spec, "language": language,
                     "resolved_model_revision": model.config._commit_hash,
                     "imbalance_method": "BCEWithLogitsLoss.pos_weight" if task == "multilabel" else "CrossEntropyLoss.weight",
                     "class_weights": dict(zip(labels, weights.tolist())), "status": "training",
                     "device": torch.cuda.get_device_name(0) if cuda else "cpu", "fp16": args.fp16,
                     "checkpoint_selection": "validation F1-macro com threshold 0.5" if task == "multilabel" else "validation F1-macro"})
    save_json(metadata, output / "run.json")
    train_result = trainer.train()
    trainer.save_state()
    scores = {}
    for name in ["validation", "test"]:
        logits = trainer.predict(data[name], metric_key_prefix=name).predictions
        scores[name] = expit(logits) if task == "multilabel" else softmax(logits, axis=1)
    reports = evaluate_scores(splits, labels, task, scores, True, output, language)
    metadata.update({"status": "completed", "duration_seconds": time.perf_counter() - start,
                     "best_checkpoint": str(Path(trainer.state.best_model_checkpoint).relative_to(output)),
                     "best_validation_f1_macro": trainer.state.best_metric, "training_metrics": train_result.metrics})
    save_json(metadata, output / "run.json")
    return reports


def main() -> None:
    """Executa o treinamento principal do BERTimbau em português."""
    parser = argparse.ArgumentParser(description="Fine-tuning do treinamento")
    parser.add_argument("--config", type=Path, default=ROOT / "Treinamento/config/treinamento.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_training(config, "pt", config["task"], args.run_name, args.allow_cpu)


if __name__ == "__main__":
    main()
