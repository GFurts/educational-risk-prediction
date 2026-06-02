"""
src/data/prepare.py
-------------------
Carrega os três CSVs anuais, padroniza colunas,
converte tipos e salva um parquet unificado.
"""

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Caminhos
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "pede_unified.parquet"

RAW_FILES = {
    2022: RAW_DIR / "PEDE2022.csv",
    2023: RAW_DIR / "PEDE2023.csv",
    2024: RAW_DIR / "PEDE2024.csv",
}


def _extrair_fase(valor):
    if pd.isna(valor):
        return None

    valor = str(valor).strip().upper()

    if valor == "ALFA":
        return 0

    # Extrai o primeiro número que encontrar
    match = re.search(r"\d+", valor)

    return int(match.group()) if match else None


def _extrair_idade(valor):
    if pd.isna(valor):
        return None

    valor = str(valor).strip()

    if "/" in valor:
        return int(valor.split("/")[1])

    try:
        return int(float(valor))
    except ValueError:
        return None


def load_year(year: int) -> pd.DataFrame:
    """Carrega e pré-processa um CSV anual."""
    path = RAW_FILES[year]
    log.info(f"Carregando {path} ...")

    df = pd.read_csv(path)
    log.info(f"Shape: {df.shape}")

    df["ano"] = year

    df["Fase"] = df["Fase"].apply(_extrair_fase)

    col_idade = "Idade" if "Idade" in df.columns else "Idade 22"
    df[col_idade] = df[col_idade].apply(_extrair_idade)

    return df


def unify(dfs: dict) -> pd.DataFrame:
    """Concatena os três anos e aplica ajustes finais."""
    unified = pd.concat(dfs.values(), ignore_index=True)

    log.info(f"Dataset unificado: {unified.shape}")
    log.info(f"Anos disponíveis: {sorted(unified['ano'].unique())}")

    return unified


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dfs = {year: load_year(year) for year in RAW_FILES}
    unified = unify(dfs)

    unified.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"✓ Salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
