"""Valida o schema do dado bruto e regenera o dicionário de dados."""

import pandas as pd

from .common import ROOT, load_params, repo_path
from .schema import CLEAN_SCHEMA, RAW_SCHEMA, data_dictionary_md


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"track_name": "name", "artists": "artist"})
    return RAW_SCHEMA.validate(df, lazy=True)  # lazy: reporta todas as violações de uma vez


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    return CLEAN_SCHEMA.validate(df, lazy=True)


def main() -> None:
    raw = pd.read_csv(repo_path(load_params()["data"]["raw"]))
    validate_raw(raw)
    (ROOT / "data" / "DATA_DICTIONARY.md").write_text(data_dictionary_md(), encoding="utf-8")
    print(f"Schema OK: {len(raw)} linhas")


if __name__ == "__main__":
    main()
