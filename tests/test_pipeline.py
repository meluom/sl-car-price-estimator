"""Automated tests run by GitHub Actions on every push (MLOps: continuous integration)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_prep import _parse_engine, clean, load_raw  # noqa: E402
import catboost_wrapper  # noqa: E402,F401
import train  # noqa: E402


# --- Data cleaning rules ---
@pytest.mark.parametrize("raw, expected", [(1.5, 1500), (18, 1800), (130, 1300), (15000, 1500), (149000, 1490), (1300, 1300)])
def test_engine_typos_are_fixed(raw, expected):
    assert _parse_engine(raw) == expected


def test_engine_invalid_becomes_missing():
    assert np.isnan(_parse_engine(0)) and np.isnan(_parse_engine(np.nan))


@pytest.fixture(scope="module")
def cleaned():
    df, audit = clean(load_raw(ROOT / "data" / "raw"), ref_year=2026)
    return df, audit


def test_clean_data_passes_validation(cleaned):
    df, _ = cleaned
    assert train.validate(df) == []


def test_no_negotiable_or_fake_prices(cleaned):
    df, audit = cleaned
    assert df["Price_Rs"].notna().all() and (df["Price_Rs"] >= 300_000).all()
    assert audit["negotiable_removed"] > 0


def test_no_duplicate_urls(cleaned):
    df, _ = cleaned
    assert not df["URL"].duplicated().any()


def test_validation_catches_broken_data():
    assert train.validate(pd.DataFrame({"Price_mn": [1.0]}))  # missing columns, too few rows


# --- Drift metric ---
def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    base = rng.normal(10, 2, 5000)
    assert train.psi(base, rng.normal(10, 2, 5000)) < 0.1
    assert train.psi(base, rng.normal(14, 2, 5000)) > 0.25


# --- Quality gate ---
def test_gate_rejects_weak_model():
    assert train.gate({"R2": 0.80, "MAPE_%": 15}, {"R2": 0.93})


def test_gate_accepts_good_model():
    assert train.gate({"R2": 0.93, "MAPE_%": 9}, {"R2": 0.929}) == []


# --- Deployed model ---
@pytest.fixture(scope="module")
def model_and_meta():
    model = joblib.load(ROOT / "models" / "car_price_model.pkl")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    return model, meta


def car(**kw):
    base = {"Brand": "TOYOTA", "Family": "TOYOTA VITZ", "Age": 10, "Mileage_km": 80000.0, "Engine_cc": 1000.0, "Cylinders": 3.0,
            "Gear": "Automatic", "Fuel": "Petrol", "Body": "Hatchback", "Condition": "Used", "Country": "Japan",
            "Location": "Colombo", "PowerSteering": 1, "PowerShutters": 1, "PowerMirrors": 1}
    base.update(kw)
    return pd.DataFrame([base])


def test_model_predicts_plausible_price(model_and_meta):
    model, meta = model_and_meta
    price = model.predict(car()[meta["features"]])[0]
    assert 3 < price < 15, f"2016 Vitz predicted at Rs {price:.2f}M"


def test_older_car_is_cheaper(model_and_meta):
    model, meta = model_and_meta
    new = model.predict(car(Age=3)[meta["features"]])[0]
    old = model.predict(car(Age=20)[meta["features"]])[0]
    assert old < new


def test_metadata_meets_quality_bar(model_and_meta):
    _, meta = model_and_meta
    assert meta["test_metrics"]["R2"] >= train.GATE["min_r2"]


# --- App smoke test ---
def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
