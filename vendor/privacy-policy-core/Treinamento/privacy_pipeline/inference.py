"""Inferência local portátil: texto segmentado para categorias OPP-115."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PrivacyClassifier:
    """Carrega todos os artefatos locais e usa os thresholds da validação."""

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        self.directory = Path(model_dir)
        verification = self.directory / "export_verification.json"
        if verification.exists() and json.loads(verification.read_text(encoding="utf-8"))["status"] != "verified":
            raise ValueError("O pacote falhou na verificação da exportação.")
        self.settings = json.loads((self.directory / "inference.json").read_text(encoding="utf-8"))
        mapping = json.loads((self.directory / "label_mapping.json").read_text(encoding="utf-8"))
        self.labels = [mapping[str(i)] for i in range(len(mapping))]
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("Mapeamento de labels inválido.")
        self.task = self.settings["task"]
        if self.task not in {"multilabel", "multiclass"}:
            raise ValueError("Tarefa inválida no pacote.")
        self.max_length = int(self.settings["max_length"])
        if self.settings["long_text_strategy"] != "head_truncation_with_audit":
            raise ValueError("Estratégia de texto longo incompatível.")
        self.thresholds = None
        if self.task == "multilabel":
            info = json.loads((self.directory / "thresholds.json").read_text(encoding="utf-8"))
            if info["labels"] != self.labels or info["score_kind"] != "probability" or info["fitted_on"] != "validation":
                raise ValueError("Thresholds incompatíveis com o modelo.")
            thresholds = torch.tensor(info["thresholds"], dtype=torch.float32)
            if thresholds.shape != (len(self.labels),) or not torch.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
                raise ValueError("Valores de threshold inválidos.")
            self.thresholds = thresholds
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True)
        self.model, loading = AutoModelForSequenceClassification.from_pretrained(
            self.directory, local_files_only=True, use_safetensors=True, output_loading_info=True)
        if any(loading.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
            raise ValueError("Checkpoint incompleto ou incompatível.")
        if [self.model.config.id2label[i] for i in range(self.model.config.num_labels)] != self.labels:
            raise ValueError("Ordem de labels diverge da configuração do modelo.")
        if not 2 < self.max_length <= self.model.config.max_position_embeddings:
            raise ValueError("max_length inválido.")
        if self.tokenizer.truncation_side != "right":
            raise ValueError("Tokenizer deve preservar o início dos trechos.")
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """Retorna rótulos, probabilidades e aviso de truncamento por trecho."""
        if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Informe uma lista de textos não vazios.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size deve ser inteiro positivo.")
        result = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            lengths = [len(ids) for ids in self.tokenizer(batch, truncation=False, padding=False, verbose=False)["input_ids"]]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            logits = self.model(**{key: value.to(self.device) for key, value in encoded.items()}).logits.float()
            probabilities = (logits.sigmoid() if self.task == "multilabel" else logits.softmax(dim=-1)).cpu()
            for offset, scores in enumerate(probabilities):
                selected = torch.where(scores >= self.thresholds)[0].tolist() if self.thresholds is not None else [int(scores.argmax())]
                result.append({"text": batch[offset], "labels": [self.labels[i] for i in selected],
                               "probabilities": {label: float(scores[i]) for i, label in enumerate(self.labels)},
                               "n_tokens": lengths[offset], "truncated": lengths[offset] > self.max_length})
        return result


def main() -> None:
    """Recebe arquivo JSON com lista de trechos; escreve JSON UTF-8."""
    parser = argparse.ArgumentParser(description="Classifica trechos com o modelo exportado")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON contendo uma lista de strings")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    texts = json.loads(args.input.read_text(encoding="utf-8"))
    results = PrivacyClassifier(args.model_dir, args.device).predict(texts, args.batch_size)
    serialized = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.resolve() == args.input.resolve():
            raise ValueError("O arquivo de saída deve ser diferente da entrada.")
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
