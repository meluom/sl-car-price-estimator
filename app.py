"""Streamlit app: Sri Lankan used-car price estimator (Riyasewana 2026) with SHAP explanations."""
import base64
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

import catboost_wrapper  # noqa: F401  (needed to unpickle a CatBoost model if it is the deployed one)

ROOT = Path(__file__).parent
st.set_page_config(page_title="Vahana Value · SL Car Price Estimator", page_icon=str(ROOT / "assets" / "car.png"), layout="wide")

MAROON, SAFFRON, TEAL, INK, MUTED = "#5C1A2E", "#E8A317", "#0F6E6E", "#241C1F", "#8A7F84"
CAR_SVG = base64.b64encode((ROOT / "assets" / "car.svg").read_bytes()).decode()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox, .stNumberInput {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 3.2rem; max-width: 1240px; }}
h1, h2, h3, .serif {{ font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.01em; }}
.hero {{ background: linear-gradient(120deg, {MAROON} 0%, #7A2440 60%, #8E3350 100%); border-radius: 22px;
        padding: 26px 32px; display: flex; align-items: center; gap: 24px; color: #fff; margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(92,26,46,.18); }}
.hero img {{ width: 78px; height: 78px; border-radius: 18px; box-shadow: 0 0 0 3px rgba(232,163,23,.55); }}
.hero .eyebrow {{ color: {SAFFRON}; font-weight: 600; font-size: .8rem; letter-spacing: .14em; text-transform: uppercase; }}
.hero h1 {{ color: #fff; margin: 2px 0 4px 0; font-size: 2.3rem; line-height: 1.1; padding: 0; }}
.hero p {{ color: #F3E6EA; margin: 0; font-size: .95rem; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: 1rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 18px !important; border: 1px solid #EDE6E9 !important;
        box-shadow: 0 4px 18px rgba(36,28,31,.05); }}
.section-title {{ font-family: 'Fraunces', serif; font-size: 1.3rem; font-weight: 700; color: {INK}; margin: 0 0 .4rem 0; }}
.price-card {{ margin-bottom: 1rem; background: #FFF8E8; border: 1px solid #F3DDA6; border-radius: 18px; padding: 22px 24px; }}
.price-label {{ color: {MUTED}; font-size: .8rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 600; }}
.price {{ font-family: 'Fraunces', serif; font-size: 3.1rem; font-weight: 700; color: {MAROON}; line-height: 1.05; margin: 4px 0; }}
.price small {{ font-size: 1.2rem; color: {INK}; font-weight: 600; }}
.range-track {{ position: relative; height: 10px; background: #F1E3BF; border-radius: 999px; margin: 18px 0 6px 0; }}
.range-fill {{ position: absolute; left: 20%; right: 20%; top: 0; bottom: 0; background: {SAFFRON}; border-radius: 999px; }}
.range-dot {{ position: absolute; left: 50%; top: -5px; width: 20px; height: 20px; margin-left: -10px; background: {MAROON};
        border: 3px solid #fff; border-radius: 50%; box-shadow: 0 1px 4px rgba(0,0,0,.25); }}
.range-labels {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .85rem; }}
.chips {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ background: #fff; border: 1px solid #EBDDE2; color: {INK}; border-radius: 999px; padding: 3px 11px; font-size: .8rem; }}
.note {{ color: {MUTED}; font-size: .82rem; margin-top: 10px; }}
.footer {{ text-align: center; color: {MUTED}; font-size: .8rem; margin-top: 2rem; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "car_price_model.pkl")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    results = pd.read_json(ROOT / "models" / "test_results.json")
    return model, meta, results


def fmt(v, dp):
    """Round half-up from the stored value (avoids 0.8845 showing as 0.884 through float rounding)."""
    return str(Decimal(str(v)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def comparison_table(results, deployed):
    """Same layout and rounding as the report: sorted by cross-validated R², deployed model first."""
    cv_path = ROOT / "reports" / "cv_results.csv"
    cv = pd.read_csv(cv_path) if cv_path.exists() else pd.DataFrame(columns=["model", "CV_R2", "CV_R2_sd"])
    cv["model"] = cv["model"].replace({"Ridge": "Ridge (linear)"})
    t = results.merge(cv[["model", "CV_R2", "CV_R2_sd"]], on="model", how="left")
    t = t.sort_values(["CV_R2", "R2"], ascending=False, na_position="last")
    return pd.DataFrame({
        "Model": [f"{m} (deployed)" if m == deployed else m for m in t["model"]],
        "CV R² (mean ± sd)": [f"{fmt(a, 3)} ± {fmt(b, 3)}" if pd.notna(a) else "–" for a, b in zip(t["CV_R2"], t["CV_R2_sd"])],
        "Test R²": [fmt(v, 3) for v in t["R2"]],
        "MAE (Rs mn)": [fmt(v, 2) for v in t["MAE_mn"]],
        "RMSE (Rs mn)": [fmt(v, 2) for v in t["RMSE_mn"]],
        "MAPE": [fmt(v, 1) + "%" for v in t["MAPE_%"]],
    })


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


def nice(name, value):
    """Readable label for a feature and its value in the drivers chart."""
    yn = {1: "yes", 0: "no"}
    labels = {
        "Age": lambda v: f"Age · {int(v)} yrs", "Mileage_km": lambda v: f"Mileage · {int(v):,} km",
        "Engine_cc": lambda v: f"Engine · {int(v):,} cc", "Cylinders": lambda v: f"Cylinders · {int(v)}",
        "Family": lambda v: f"Model · {str(v).title()}", "Brand": lambda v: f"Brand · {str(v).title()}",
        "PowerSteering": lambda v: f"Power steering · {yn.get(int(v), v)}", "PowerShutters": lambda v: f"Power shutters · {yn.get(int(v), v)}",
        "PowerMirrors": lambda v: f"Power mirrors · {yn.get(int(v), v)}",
    }
    return labels.get(name, lambda v: f"{name} · {v}")(value)


def drivers_chart(sv, top=8):
    vals, data, names = sv.values[0], sv.data[0], sv.feature_names
    order = np.argsort(-np.abs(vals))[:top][::-1]
    pct = [(np.exp(vals[i]) - 1) * 100 for i in order]
    labels = [nice(names[i], data[i]) for i in order]
    fig, ax = plt.subplots(figsize=(7, 0.48 * len(order) + 0.6))
    colors = [TEAL if p >= 0 else MAROON for p in pct]
    ax.barh(range(len(order)), pct, color=colors, height=0.62)
    for y, p in enumerate(pct):
        ax.text(p + (1 if p >= 0 else -1), y, f"{p:+.0f}%", va="center", ha="left" if p >= 0 else "right",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(range(len(order)), labels, fontsize=10, color=INK)
    ax.axvline(0, color="#CFC6CA", lw=1)
    lim = max(abs(min(pct)), abs(max(pct))) * 1.35 + 2
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Effect on price compared with an average car", fontsize=9, color=MUTED)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#E3DBDE")
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.tight_layout()
    return fig


# ── Header ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <img src="data:image/svg+xml;base64,{CAR_SVG}" alt="car icon"/>
  <div>
    <div class="eyebrow">Vahana Value · Riyasewana 2026 · Explainable ML</div>
    <h1>What is this car really worth?</h1>
    <p>Fair asking-price estimates for Sri Lankan used cars, trained on {meta['n_rows']:,} adverts
       ({meta['date_range'][0]} to {meta['date_range'][1]}), with the reasons behind every estimate.</p>
  </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_model, tab_about = st.tabs(["Price a car", "How the model works", "About the data"])

with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Your car</div>', unsafe_allow_html=True)
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
        lo, hi = pred * (1 - e), pred * (1 + e)
        model_name = family.replace(brand + " ", "").title()
        st.markdown(f"""
<div class="price-card">
  <div class="price-label">Estimated asking price</div>
  <div class="price">Rs {pred:,.2f} <small>million</small></div>
  <div class="range-track"><div class="range-fill"></div><div class="range-dot"></div></div>
  <div class="range-labels"><span>Rs {lo:,.2f}M</span><span>likely range</span><span>Rs {hi:,.2f}M</span></div>
  <div class="chips">
    <span class="chip">{brand.title()} {model_name}</span><span class="chip">{int(yom)}</span>
    <span class="chip">{int(mileage):,} km</span><span class="chip">{fuel}</span><span class="chip">{condition}</span><span class="chip">{location}</span>
  </div>
  <div class="note">Half of unseen test cars were priced within ±{e:.1%} of their real asking price.</div>
</div>
""", unsafe_allow_html=True)
        if info["n"] < 15:
            st.warning(f"Only {info['n']} adverts for this model in the training data, so treat this estimate with extra caution.")

        with st.container(border=True):
            st.markdown('<div class="section-title">What drives this price</div>', unsafe_allow_html=True)
            st.caption("SHAP explanation: how much each detail moves this car's price up (teal) or down (maroon) "
                       "compared with an average car in the data.")
            st.pyplot(drivers_chart(explain(row)), clear_figure=True)

with tab_model:
    with st.container(border=True):
        st.markdown('<div class="section-title">Model comparison on 997 unseen test cars</div>', unsafe_allow_html=True)
        st.dataframe(comparison_table(results, meta["model_name"]), width="stretch", hide_index=True)
        st.caption(f"Deployed model: **{meta['model_name']}**, chosen on cross-validated R² (the most stable across folds), not on the test score.")
    fig_dir = ROOT / "reports" / "figures"
    charts = [("Model comparison", "07_model_comparison.png"),
              ("Global feature importance (SHAP)", "09_shap_bar.png"),
              ("Direction of feature effects (SHAP beeswarm)", "10_shap_beeswarm.png"),
              ("How age and mileage affect price (SHAP)", "11_shap_age_mileage.png"),
              ("LIME explanation of one car", "13_lime.png")]
    for title, f in charts:
        if (fig_dir / f).exists():
            with st.container(border=True):
                st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
                st.image(str(fig_dir / f), width="stretch")

with tab_about:
    with st.container(border=True):
        st.markdown(f"""
<div class="section-title">About the data</div>

**Source.** {meta['n_rows']:,} priced car adverts from **Riyasewana.com**, posted {meta['date_range'][0]} to {meta['date_range'][1]},
covering about 30 popular model families. The raw listings come from the
[Sri-Lanka-Car-Market-Analysis-2026](https://github.com/savindu777/Sri-Lanka-Car-Market-Anlaysis-2026) repository (MIT licence).
All cleaning and modelling are original.

**Data quality decisions**
- Adverts marked "Negotiable" (22%) have no price and were removed.
- Engine cc typing errors were corrected (e.g. 15,000 → 1,500; 1.5 → 1,500).
- Implausible prices, such as deposits and lease instalments, were removed, along with prices more than 4× away from similar cars.
- Re-posted adverts were kept together in a **grouped train/test split** to prevent leakage.
- Prices are modelled on a log scale.

**Limitations.** These are asking prices, not sale prices. The data covers four months of 2026 and does not model trim levels. Rare models are less accurate.
""", unsafe_allow_html=True)

st.markdown(f'<div class="footer">Model {meta.get("model_version", "v1")} · test R² {fmt(meta["test_metrics"]["R2"], 3)} · '
            'Melanie Samaratunga · 269188K · MSc in AI, University of Moratuwa · '
            '<a href="https://github.com/meluom/sl-car-price-estimator">Source on GitHub</a></div>', unsafe_allow_html=True)
