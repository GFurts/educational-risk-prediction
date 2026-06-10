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

    # Pega o INDE e Pedra do ano correto antes do rename
    inde_col = {2022: "INDE 22", 2023: "INDE 2023", 2024: "INDE 2024"}
    pedra_col = {2022: "Pedra 22", 2023: "Pedra 2023", 2024: "Pedra 2024"}
    df["inde"] = df[inde_col[year]]
    df["pedra"] = df[pedra_col[year]]

    # Padroniza nomes das colunas entre os anos
    RENAME = {
        "Nome": "nome",
        "Nome Anonimizado": "nome",
        "Idade 22": "idade",
        "Idade": "idade",
        "Ano nasc": "ano_nasc",
        "Data de Nasc": "data_nasc",
        "Gênero": "genero",
        "Ano ingresso": "ano_ingresso",
        "Instituição de ensino": "instituicao",
        "Fase Ideal": "fase_ideal",
        "Fase ideal": "fase_ideal",
        "Defas": "defasagem",
        "Defasagem": "defasagem",
        "Matem": "nota_mat",
        "Mat": "nota_mat",
        "Portug": "nota_por",
        "Por": "nota_por",
        "Inglês": "nota_ing",
        "Ing": "nota_ing",
        "Atingiu PV": "ponto_virada",
        "IAA": "iaa",
        "IEG": "ieg",
        "IPS": "ips",
        "IDA": "ida",
        "IPP": "ipp",
        "IPV": "ipv",
        "IAN": "ian",
    }
    df = df.rename(columns=RENAME)

    # Remove colunas desnecessárias
    COLUNAS_REMOVER = [
        "INDE 22",
        "INDE 2023",
        "INDE 2024",
        "Pedra 20",
        "Pedra 21",
        "Pedra 22",
        "Pedra 2023",
        "Pedra 2024",
        "Pedra 23",
        "Cg",
        "Cf",
        "Ct",
        "Nº Av",
        "Avaliador1",
        "Avaliador2",
        "Avaliador3",
        "Avaliador4",
        "Avaliador5",
        "Avaliador6",
        "Rec Av1",
        "Rec Av2",
        "Rec Av3",
        "Rec Av4",
        "Rec Psicologia",
        "Indicado",
        "Destaque IEG",
        "Destaque IDA",
        "Destaque IPV",
        "Destaque IPV.1",
        "INDE 23",
        "Escola",
        "Ativo/ Inativo",
        "Ativo/ Inativo.1",
    ]
    df = df.drop(columns=[c for c in COLUNAS_REMOVER if c in df.columns])

    log.info(f"Shape: {df.shape}")

    df["ano"] = year
    df["Fase"] = df["Fase"].apply(_extrair_fase)

    col_idade = "idade" if "idade" in df.columns else "idade"
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
