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

    # Pega INDE e Pedra do ano correto
    inde_col = {2022: "INDE 22", 2023: "INDE 2023", 2024: "INDE 2024"}
    pedra_col = {2022: "Pedra 22", 2023: "Pedra 2023", 2024: "Pedra 2024"}
    df["inde"] = df[inde_col[year]]
    df["pedra"] = df[pedra_col[year]]

    # Padroniza nomes das colunas entre os anos
