"""Load and clean raw Riyasewana car listings.

Works on the per-model CSV files from the source repository (data/raw/) and on any
freshly scraped CSVs with the same columns, so the pipeline can be re-run on new data.
Every cleaning rule records how many rows it touched in an audit log, for the report.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

LUXURY = {"MERCEDES-BENZ", "BMW", "AUDI", "JAGUAR", "VOLVO"}


def load_raw(folder="data/raw"):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    df = pd.concat([pd.read_csv(f).assign(source_file=os.path.basename(f)) for f in files], ignore_index=True)
    return df


def _family_from_file(fname):
    """Single-model files (e.g. riyasewana_toyota_aqua_*.csv) give the model family directly."""
    m = re.match(r"riyasewana_([a-z\-]+?)_([a-z\-_]+?)(?:_with_country|_edited|_cleaned|_sedan_only)*\.csv$", fname)
    if not m:
        return None
    rest = m.group(2).replace("_edited", "").replace("_cleaned", "").replace("_with_country", "")
    if rest in ("", "benz", "sedan_only") or m.group(2).startswith(("edited", "cleaned", "with_country", "sedan")):
        return None
    return rest.replace("_", " ").upper()


def _parse_engine(e):
    """Fix common typing errors in engine cc: missing/extra zeros and litres entered as cc."""
    if pd.isna(e) or e <= 0:
        return np.nan
    if e < 10:          # 1.5 -> litres
        e *= 1000
    elif e < 100:       # 18 -> 1800
        e *= 100
    elif e < 600:       # 130 -> 1300
        e *= 10
    while e > 7000:     # 15000 -> 1500, 149000 -> 1490
        e /= 10
    return round(e) if 600 <= e <= 7000 else np.nan


def clean(raw: pd.DataFrame, ref_year: int = 2026):
    log = {"raw_rows": len(raw)}
    df = raw.copy()

    # 1. Merge duplicated columns created by inconsistent scraping (singular/plural headers)
    for a, b in [("Number of Cylinders", "Number of Cylinder"), ("Power Shutters", "Power Shutter"), ("Power Mirrors", "Power Mirror")]:
        if b in df:
            log[f"merged_{b}"] = int(df[a].isna().sum())
            df[a] = df[a].fillna(df[b]); df = df.drop(columns=b)

    # 2. Remove duplicate adverts (same URL)
    n = len(df); df = df.drop_duplicates(subset="URL"); log["dup_url_removed"] = n - len(df)

    # 3. Parse price; 'Negotiable' adverts have no target and are removed
    price = df["Price"].astype(str).str.replace("Rs.", "", regex=False).str.replace(r"[^0-9]", "", regex=True)
    df["Price_Rs"] = pd.to_numeric(price, errors="coerce")
    log["negotiable_removed"] = int(df["Price_Rs"].isna().sum())
    df = df[df["Price_Rs"].notna()]

    # 4. Parse mileage, date, engine
    df["Mileage_km"] = pd.to_numeric(df["Mileage"].astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce")
    df["Ad_Date"] = pd.to_datetime(df["Ad Date"].astype(str).str.split(",").str[0], format="%Y %b %d", errors="coerce")
    eng_raw = df["Engine (cc)"]
    df["Engine_cc"] = eng_raw.apply(_parse_engine)
    log["engine_corrected"] = int(((df["Engine_cc"] != eng_raw) & df["Engine_cc"].notna()).sum())
    log["engine_set_missing"] = int((df["Engine_cc"].isna() & eng_raw.notna()).sum())

    # 5. Standardise categories
    df["Brand"] = df["Brand"].str.strip().str.upper()
    df["Location"] = df["Location"].astype(str).str.strip().str.title()
    df["Condition"] = df["Condition"].str.extract(r"(Used|Recondition|Brand New|Antique)", expand=False)
    df["Fuel"] = df["Fuel Type"]; df["Body"] = df["Body Type"]; df["Country"] = df["Manufacturing Country"]
    df["Cylinders"] = df["Number of Cylinders"].replace(0, np.nan)
    for c, new in [("Power Steering", "PowerSteering"), ("Power Shutters", "PowerShutters"), ("Power Mirrors", "PowerMirrors")]:
        df[new] = df[c].map({"Yes": 1, "No": 0})

    # 6. Antique cars form a separate collector market
    log["antique_removed"] = int((df["Condition"] == "Antique").sum())
    df = df[df["Condition"] != "Antique"]

    # 7. Model family: from the file name for single-model files, else the first word of the model
    fam_file = df["source_file"].map(_family_from_file)
    first = df["Model"].astype(str).str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True).str.split().str[0]
    df["Family"] = fam_file.fillna(first)
    df.loc[df["Family"].isin(["WAGON", "WAGON R"]), "Family"] = "WAGON R"
    df["Family"] = df["Family"].str.replace("CR-V", "CRV")
    df["Family"] = df["Brand"] + " " + df["Family"]
    counts = df["Family"].value_counts()
    rare = df["Family"].map(counts) < 5
    df.loc[rare, "Family"] = df.loc[rare, "Brand"] + " OTHER"
    log["rare_families_grouped"] = int(rare.sum())

    # 8. Mileage below 1,000 km on a registered used car older than 2 years is a typing error
    df["Age"] = (ref_year - df["YOM"]).clip(lower=0)
    bad_mil = (df["Mileage_km"] < 1000) & (df["Condition"] == "Used") & (df["Age"] > 2)
    df.loc[bad_mil, "Mileage_km"] = np.nan
    log["mileage_set_missing"] = int(bad_mil.sum())

    # 9. Price sanity: below Rs 300k is not a car price (deposit / monthly lease / typo)
    low = df["Price_Rs"] < 300_000
    log["price_too_low_removed"] = int(low.sum()); df = df[~low]

    # 10. Price outliers relative to similar cars (same family, YOM within ±2):
    #     more than 4x above or below the peer median is almost certainly a typo
    df = df.reset_index(drop=True)
    lp = np.log(df["Price_Rs"])
    peer_med = np.array([np.median(lp[(df["Family"] == f) & ((df["YOM"] - y).abs() <= 2)]) for f, y in zip(df["Family"], df["YOM"])])
    out = np.abs(lp - peer_med) > np.log(4)
    log["price_peer_outliers_removed"] = int(out.sum())
    log["peer_outlier_examples"] = df.loc[out, ["Brand", "Model", "YOM", "Price_Rs"]].head(8).to_dict("records")
    df = df[~out].reset_index(drop=True)

    df["Price_mn"] = df["Price_Rs"] / 1e6
    # Re-posts: same car advertised more than once (identical specs, mileage and price)
    df["listing_group"] = df.groupby(["Brand", "Model", "YOM", "Mileage_km", "Price_Rs"], dropna=False, sort=False).ngroup()
    log["final_rows"] = len(df)
    log["repost_rows"] = int(len(df) - df["listing_group"].nunique())

    keep = ["Brand", "Family", "Model", "YOM", "Age", "Mileage_km", "Engine_cc", "Cylinders", "Gear", "Fuel", "Body",
            "Condition", "Country", "Location", "PowerSteering", "PowerShutters", "PowerMirrors",
            "Ad_Date", "URL", "Price_Rs", "Price_mn", "listing_group"]
    return df[keep], log


if __name__ == "__main__":
    d, log = clean(load_raw())
    for k, v in log.items():
        print(k, v)
    d.to_csv("data/cars_clean.csv", index=False)
