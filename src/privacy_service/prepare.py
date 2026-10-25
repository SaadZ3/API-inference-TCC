"""CLI P3: arquivo PDF/TXT para JSON, sem classificar nem calcular score."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from .extraction import ExtractionLimits, IngestionError, SCOPES, extract_path
from .segmentation import PolicySegmenter


def main() -> int:
    # JSON de erro usa UTF-8 também quando redirecionado no Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser(description="Extrai e segmenta uma política PDF/TXT para o modelo local.")
    parser.add_argument("--input", required=True, type=Path, help="PDF digital ou TXT UTF-8.")
    parser.add_argument("--output", required=True, type=Path, help="Novo arquivo JSON; não sobrescreve arquivos.")
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("MODEL_DIR"))
    parser.add_argument("--document-scope", required=True, choices=sorted(SCOPES))
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("Informe --model-dir ou configure MODEL_DIR.")
    if args.output.resolve() == args.input.resolve() or args.output.exists():
        parser.error("A saída deve ser um arquivo novo, diferente da entrada.")
    try:
        document = extract_path(args.input, document_scope=args.document_scope, limits=ExtractionLimits())
        result = PolicySegmenter(args.model_dir).prepare(document)
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        # Escrita exclusiva evita sobrescrita, inclusive se outra execução criou a saída.
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        print(f"Preparação concluída: {len(result['segments'])} trechos; extração {result['extraction']['status']}.")
        if result["extraction"]["warnings"]:
            print(f"Avisos registrados no JSON: {len(result['extraction']['warnings'])}.")
        return 0
    except IngestionError as error:
        print(json.dumps({"error": {"code": error.code, "message": error.message}}, ensure_ascii=False), file=sys.stderr)
        return 1
    except OSError:
        print('{"error":{"code":"INVALID_INPUT","message":"Não foi possível gravar a saída. Confira a pasta e as permissões."}}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
