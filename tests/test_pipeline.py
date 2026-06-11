"""
tests/test_pipeline.py
----------------------
Testes do pipeline de dados e do modelo preditivo.
"""

from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).parent.parent


def test_model_loads():
    """Modelo carrega sem erros."""
    model = joblib.load(BASE_DIR / "models/risk_model.joblib")
    assert model is not None


def test_features_loads():
    """Lista de features carrega corretamente."""
    features = joblib.load(BASE_DIR / "models/features.joblib")
    assert isinstance(features, list)
    assert len(features) == 7


def test_threshold_loads():
    """Threshold carrega e está no intervalo esperado."""
    threshold = joblib.load(BASE_DIR / "models/threshold.joblib")
    assert 0 < threshold < 1


def test_model_predicts_probability():
    """Modelo retorna probabilidade entre 0 e 1."""
    model = joblib.load(BASE_DIR / "models/risk_model.joblib")
    entrada = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2023]])
    prob = model.predict_proba(entrada)[0][1]
    assert 0.0 <= prob <= 1.0


def test_model_predicts_high_risk():
    """Aluno com indicadores baixos deve ter probabilidade de risco elevada."""
    model = joblib.load(BASE_DIR / "models/risk_model.joblib")
    entrada = np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2022]])
    prob = model.predict_proba(entrada)[0][1]
    assert prob >= 0.3


def test_model_predicts_low_risk():
    """Aluno com indicadores altos deve ter baixa probabilidade de risco."""
    model = joblib.load(BASE_DIR / "models/risk_model.joblib")
    threshold = joblib.load(BASE_DIR / "models/threshold.joblib")
    entrada = np.array([[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 2024]])
    prob = model.predict_proba(entrada)[0][1]
    assert prob < threshold
