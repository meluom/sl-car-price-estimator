"""Streamlit app: Sri Lankan used-car price estimator (Riyasewana 2026) with SHAP explanations."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

import catboost_wrapper  # noqa: F401  (needed to unpickle a CatBoost model if it is the deployed one)

ROOT = Path(__file__).parent
st.set_page_config(page_title="SL Car Price Estimator", page_icon="🚗", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "car_price_model.pkl")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    results = pd.read_json(ROOT / "models" / "test_results.json")
    return model, meta, results


@st.cache_resource
def get_explainer(_model, is_catboost):
    inner = _model.regressor_
    return shap.TreeExplainer(inner.model_ if is_catboost else inner.named_steps["model"])


model, meta, results = load()
IS_CB = meta["model_name"] == "CatBoost"
FAM = meta["families"]


def explain(row):
    explainer = get_explainer(model, IS_CB)
    if IS_CB:
        return explainer(row)
    pre = model.regressor_.named_steps["pre"]
    return explainer(pd.DataFrame(pre.transform(row), columns=pre.get_feature_names_out()))


def pick(label, options, default, key, col=st):
    options = list(options)
    return col.selectbox(label, options, index=options.index(default) if default in options else 0, key=key)


st.title("🚗 Sri Lankan Used Car Price Estimator")
st.caption(f"Trained on {meta['n_rows']:,} Riyasewana.com adverts ({meta['date_range'][0]} to {meta['date_range'][1]}). "
           f"Model: **{meta['model_name']}**. Estimates are advertised asking prices, not final sale prices.")

tab_predict, tab_model, tab_about = st.tabs(["Estimate a price", "How the model works", "About the data"])

with tab_predict:
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("Car details")
        brands = sorted({v["Brand"] for v in FAM.values()})
        c1, c2 = st.columns(2)
        brand = pick("Brand", brands, "TOYOTA", "brand", c1)
        fams = sorted([f for f, v in FAM.items() if v["Brand"] == brand], key=lambda f: -FAM[f]["n"])
        family = c2.selectbox("Model", fams, format_func=lambda f: f.replace(brand + " ", "").title(), key="family")
        info = FAM[family]
        k = family  # widget keys depend on the model, so defaults refresh when the model changes

        c3, c4 = st.columns(2)
        yom = c3.number_input("Year of manufacture", meta["yom_range"][0], meta["ref_year"], 2016, key="yom")
        mileage = c4.number_input("Mileage (km)", 0, 1_000_000, 80_000, step=5_000, key="mileage")
        c5, c6 = st.columns(2)
        condition = pick("Condition", meta["options"]["Condition"], "Used", "cond", c5)
        fuel = pick("Fuel", meta["options"]["Fuel"], info["Fuel"], f"fuel_{k}", c6)
        c7, c8 = st.columns(2)
        gear = pick("Transmission", meta["options"]["Gear"], info["Gear"], f"gear_{k}", c7)
        body = pick("Body type", meta["options"]["Body"], info["Body"], f"body_{k}", c8)
        c9, c10 = st.columns(2)
        engine = c9.number_input("Engine (cc)", 600, 7000, int(info["Engine"] or 1300), step=50, key=f"eng_{k}")
        location = pick("Location", meta["locations"], "Colombo", "loc", c10)
        with st.expander("More details"):
            d1, d2 = st.columns(2)
            cyl = d1.number_input("Cylinders", 2, 12, int(info["Cyl"] or 4), key=f"cyl_{k}")
            country = pick("Country of manufacture", meta["options"]["Country"], info["Country"], f"cty_{k}", d2)
            f1, f2, f3 = st.columns(3)
            ps = f1.checkbox("Power steering", True)
            pw = f2.checkbox("Power shutters", True)
            pm = f3.checkbox("Power mirrors", True)

    row = pd.DataFrame([{
        "Brand": brand, "Family": family, "Age": max(meta["ref_year"] - int(yom), 0), "Mileage_km": float(mileage),
        "Engine_cc": float(engine), "Cylinders": float(cyl), "Gear": gear, "Fuel": fuel, "Body": body, "Condition": condition,
        "Country": country, "Location": location, "PowerSteering": int(ps), "PowerShutters": int(pw), "PowerMirrors": int(pm),
    }])[meta["features"]]

    with right:
        pred = float(model.predict(row)[0])
        e = meta["median_abs_pct_err"] / 100
        st.subheader("Estimated asking price")
        st.metric("Estimate", f"Rs {pred:,.2f} million")
        st.caption(f"Half of unseen test cars were predicted within ±{e:.0%} of their asking price, "
                   f"so a likely range is **Rs {pred * (1 - e):,.2f}M – Rs {pred * (1 + e):,.2f}M**.")
        if info["n"] < 15:
            st.warning(f"Only {info['n']} adverts for this model in the training data, so treat this estimate with extra caution.")

        st.subheader("Why this price? (SHAP)")
        sv = explain(row)
        fig = plt.figure(figsize=(7, 4.8))
        shap.plots.waterfall(sv[0], max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Each bar shows how a feature pushes this estimate up (red) or down (blue) from the average car, "
                   "in log-price units: +0.10 means about +10% on price.")

with tab_model:
    st.subheader("Model comparison on the held-out test set")
    show = results.set_index("model")[["R2", "MAE_mn", "RMSE_mn", "MAPE_%"]].rename(
        columns={"MAE_mn": "MAE (Rs mn)", "RMSE_mn": "RMSE (Rs mn)", "MAPE_%": "MAPE (%)"})
    st.dataframe(show.style.format("{:.3f}").highlight_max(subset=["R2"], color="#c6efce"), width="stretch")
    fig_dir = ROOT / "reports" / "figures"
    for title, f in [("Model comparison", "07_model_comparison.png"),
                     ("Global feature importance (SHAP)", "09_shap_bar.png"),
                     ("Direction of feature effects (SHAP beeswarm)", "10_shap_beeswarm.png"),
                     ("How age and mileage affect price (SHAP)", "11_shap_age_mileage.png"),
                     ("LIME explanation of one car", "13_lime.png")]:
        if (fig_dir / f).exists():
            st.markdown(f"**{title}**")
            st.image(str(fig_dir / f), width="stretch")

with tab_about:
    st.markdown(f"""
**Source.** {meta['n_rows']:,} priced car adverts from **Riyasewana.com**, posted {meta['date_range'][0]} to {meta['date_range'][1]}
and covering about 30 popular model families. The raw listings come from the
[Sri-Lanka-Car-Market-Analysis-2026](https://github.com/savindu777/Sri-Lanka-Car-Market-Anlaysis-2026) repository (MIT licence).
All cleaning and modelling are original.

**Data quality decisions**
- Adverts marked "Negotiable" (22%) have no price and were removed.
- Engine cc typing errors were corrected (e.g. 15,000 → 1,500; 1.5 → 1,500).
- Implausible prices, such as deposits and lease instalments, were removed, along with prices more than 4× away from similar cars.
- Re-posted adverts were kept together in a **grouped train/test split** to prevent leakage.
- Prices are modelled on a log scale.

**Limitations.** These are asking prices, not sale prices. The data covers four months of 2026 and does not model trim levels. Rare models are less accurate.
""")
