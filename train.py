"""Automated retraining pipeline with a quality gate and model registry (MLOps).

Run:  python train.py            # retrain, evaluate, promote only if the gate passes
      python train.py --dry-run  # retrain and evaluate, but never replace the live model

Steps
  1. Load raw adverts from data/raw and clean them with data_prep.clean (same rules as the notebook).
  2. Data validation: schema, row count, price range. The run stops if the data looks broken.
  3. Drift check: compares the new data with the data the live model was trained on (PSI).
  4. Train CatBoost with the tuned settings from the notebook, on the same leakage-safe grouped split.
  5. Quality gate: promote only if the test R² and MAPE meet fixed thresholds AND the model is no
     worse than the one in production.
  6. On promotion: refit on all data, save models/car_price_model.pkl and metadata.json, and append
     a versioned entry to models/registry.json. Streamlit Cloud redeploys automatically on push.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from catboost_wrapper import CatBoostReg
from data_prep import clean, load_raw

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
FEATURES = ["Brand", "Family", "Age", "Mileage_km", "Engine_cc", "Cylinders", "Gear", "Fuel", "Body", "Condition",
            "Country", "Location", "PowerSteering", "PowerShutters", "PowerMirrors"]
CATS = ["Brand", "Family", "Location", "Gear", "Fuel", "Body", "Condition", "Country"]
PARAMS = {"iterations": 1000, "learning_rate": 0.1, "depth": 4, "l2_leaf_reg": 6}   # tuned in the notebook
SEED = 42

GATE = {"min_r2": 0.90, "max_mape": 12.0, "max_r2_drop_vs_prod": 0.02}
MIN_ROWS = 3000
DRIFT_COLS = ["Age", "Mileage_km", "Engine_cc", "Price_mn"]


def data_hash(folder):
    h = hashlib.sha256()
    for f in sorted(Path(folder).glob("*.csv")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:12]


def validate(df):
    """Fail fast if the cleaned data is not fit for training."""
    problems = []
    missing = [c for c in FEATURES + ["Price_mn", "listing_group"] if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows (minimum {MIN_ROWS})")
    if not df.empty and "Price_mn" in df and not df["Price_mn"].between(0.3, 500).all():
        problems.append("prices outside Rs 0.3M – 500M after cleaning")
    if not df.empty and "Age" in df and (df["Age"] < 0).any():
        problems.append("negative car ages")
    return problems


def psi(expected, actual, bins=10):
    """Population Stability Index: < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 major shift."""
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(df):
    ref_path = MODELS / "reference_stats.json"
    if not ref_path.exists():
        return {}
    ref = json.loads(ref_path.read_text())
    return {c: round(psi(ref[c], df[c]), 4) for c in DRIFT_COLS if c in ref}


def build_model():
    return TransformedTargetRegressor(regressor=CatBoostReg(cat_features=CATS, random_seed=SEED, thread_count=-1, **PARAMS),
                                      func=np.log, inverse_func=np.exp)


def evaluate(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["listing_group"]))
    m = build_model().fit(df.iloc[tr][FEATURES], df.iloc[tr]["Price_mn"])
    y, p = df.iloc[te]["Price_mn"], m.predict(df.iloc[te][FEATURES])
    err = np.abs(p - y) / y
    return {"R2": round(r2_score(y, p), 4), "MAE_mn": round(mean_absolute_error(y, p), 4),
            "MAPE_%": round(100 * mean_absolute_percentage_error(y, p), 4),
            "median_abs_pct_err": round(100 * float(np.median(err)), 2),
            "within_20pct": round(float((err <= 0.2).mean()), 4), "n_test": int(len(te))}


def gate(metrics, prod):
    reasons = []
    if metrics["R2"] < GATE["min_r2"]:
        reasons.append(f"R² {metrics['R2']} < {GATE['min_r2']}")
    if metrics["MAPE_%"] > GATE["max_mape"]:
        reasons.append(f"MAPE {metrics['MAPE_%']}% > {GATE['max_mape']}%")
    if prod and metrics["R2"] < prod.get("R2", 0) - GATE["max_r2_drop_vs_prod"]:
        reasons.append(f"R² {metrics['R2']} is more than {GATE['max_r2_drop_vs_prod']} below production ({prod['R2']})")
    return reasons


def main(dry_run=False):
    raw = load_raw(ROOT / "data" / "raw")
    df, audit = clean(raw, ref_year=2026)
    print(f"[data] {len(raw):,} raw adverts → {len(df):,} clean rows")

    problems = validate(df)
    if problems:
        print("[validate] FAILED:", "; ".join(problems)); sys.exit(1)
    print("[validate] passed")

    drift = drift_report(df)
    if drift:
        flag = {c: v for c, v in drift.items() if v > 0.25}
        print(f"[drift] PSI {drift}" + (f"  ⚠ major shift in {list(flag)}" if flag else "  (stable)"))

    metrics = evaluate(df)
    print(f"[evaluate] {metrics}")
    meta_path = MODELS / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    prod = meta.get("test_metrics", {})
    reasons = gate(metrics, prod)
    if reasons:
        print("[gate] REJECTED, live model kept:", "; ".join(reasons)); sys.exit(2)
    print(f"[gate] passed (production R² {prod.get('R2', 'n/a')})")
    if dry_run:
        print("[dry-run] not promoting"); return

    final = build_model().fit(df[FEATURES], df["Price_mn"])
    joblib.dump(final, MODELS / "car_price_model.pkl", compress=3)

    registry_path = MODELS / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    version = f"v{len(registry) + 1}"
    fam = df.groupby("Family").agg(Brand=("Brand", "first"), Engine=("Engine_cc", "median"), Fuel=("Fuel", lambda s: s.mode()[0]),
                                   Body=("Body", lambda s: s.mode()[0]), Country=("Country", lambda s: s.mode()[0]),
                                   Gear=("Gear", lambda s: s.mode()[0]), Cyl=("Cylinders", "median"), n=("Price_mn", "size"))
    meta.update({
        "model_name": "CatBoost", "model_version": version, "best_params": PARAMS, "features": FEATURES, "ref_year": 2026,
        "test_metrics": {k: metrics[k] for k in ("R2", "MAE_mn", "MAPE_%")}, "median_abs_pct_err": metrics["median_abs_pct_err"],
        "families": {f: {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in r.items()}
                     for f, r in fam.to_dict("index").items()},
        "locations": df["Location"].value_counts().index.tolist(),
        "options": {c: sorted(df[c].dropna().unique().tolist()) for c in ["Gear", "Fuel", "Body", "Condition", "Country"]},
        "yom_range": [int(df.YOM.min()), int(df.YOM.max())], "n_rows": int(len(df)),
        "date_range": [str(df.Ad_Date.min().date()), str(df.Ad_Date.max().date())],
    })
    meta_path.write_text(json.dumps(meta, indent=1))
    ref = {c: df[c].dropna().sample(min(2000, df[c].notna().sum()), random_state=SEED).round(3).tolist() for c in DRIFT_COLS}
    (MODELS / "reference_stats.json").write_text(json.dumps(ref))
    registry.append({"version": version, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "data_hash": data_hash(ROOT / "data" / "raw"), "rows": int(len(df)), "metrics": metrics,
                     "drift_psi": drift, "params": PARAMS})
    registry_path.write_text(json.dumps(registry, indent=1))
    print(f"[promote] {version} saved: models/car_price_model.pkl, metadata.json, registry.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(**vars(ap.parse_args()))
