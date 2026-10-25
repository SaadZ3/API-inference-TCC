"""Gera um único pacote com os arquivos necessários para treinar no Colab."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import yaml
from tratamento.cli import ROOT

def main() -> None:
    config = yaml.safe_load((ROOT / "Treinamento/config/treinamento.yaml").read_text(encoding="utf-8"))
    paths = [*sorted((ROOT / "Tratamento/tratamento").glob("*.py")),
             *sorted((ROOT / "Treinamento/privacy_pipeline").glob("*.py")),
             ROOT / "pyproject.toml", ROOT / "requirements.txt",
             ROOT / "Tratamento/requirements.txt", ROOT / "Treinamento/requirements.txt",
             ROOT / "Treinamento/requirements-inference.txt",
             ROOT / config["treatment_config"], ROOT / "Treinamento/config/treinamento.yaml",
             ROOT / "Treinamento/config/exportacao.yaml",
             ROOT / "Tratamento/dados/manifest.json", ROOT / "Tratamento/dados/segments.jsonl",
             ROOT / config["translated_file"], ROOT / config["human_review_record"],
             ROOT / config["human_review_file"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes: {missing}")
    destination = ROOT / "Treinamento/opp115_colab.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(destination)

if __name__ == "__main__":
    main()
