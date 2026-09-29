"""
Script 1/5 - Traffic incident categorization
Project: Calgary Traffic Collision Risk Prediction

Uses FIXED absolute paths, so this script can be run from any folder
and will always read/write to the correct location.
"""

import pandas as pd
import re
import os

# ============================================================
# FIXED PROJECT PATHS - do not change regardless of where you run this from
# ============================================================
BASE_DIR = r"C:\Portafolio\calgary-traffic-risk"
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "traffic_incidents_raw.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "traffic_incidents_categorized.csv")

# --- Check the input file exists before continuing ---
if not os.path.exists(RAW_CSV):
    raise SystemExit(
        f"\nERROR: file not found at:\n  {RAW_CSV}\n"
        f"Make sure the original CSV is there, with that exact name.\n"
    )

df = pd.read_csv(RAW_CSV)
print(f"File loaded successfully: {RAW_CSV}")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}\n")


def clean(text):
    return "" if pd.isna(text) else str(text).lower()


def categorize(text):
    t = clean(text)
    if "pedestrian" in t:
        return "Pedestrian"
    if "cyclist" in t:
        return "Cyclist"
    if re.search(r"\b(two|multi|multiple|[2-9])\s*[-\s]?vehicle", t):
        return "Multi_vehicle_collision"
    if "single vehicle" in t:
        return "Single_vehicle_collision"
    if "stalled" in t or "disabled vehicle" in t:
        return "Stalled_vehicle"
    if "signal" in t and ("flashing" in t or "blank" in t or "malfunction" in t or "not working" in t):
        return "Signal_malfunction"
    if t.strip().startswith("traffic incident"):
        return "Generic_incident_no_detail"
    if "closed" in t or "reopened" in t:
        return "Road_closure"
    if "work in progress" in t or "work zone" in t:
        return "Work_zone"
    if t.strip() == "":
        return "No_description"
    return "Other_unclassified"


df["category"] = df["DESCRIPTION"].apply(categorize)
df["lane_blocked"] = df["DESCRIPTION"].apply(lambda t: "blocking" in clean(t))


def extract_vehicle_count(text):
    t = clean(text)
    if "single vehicle" in t:
        return 1
    if "two vehicle" in t or "2 vehicle" in t:
        return 2
    m = re.search(r"(\d+)\s*vehicle", t)
    if m:
        return int(m.group(1))
    if "multi-vehicle" in t or "multi vehicle" in t or "multiple vehicle" in t:
        return 3
    return None


df["vehicle_count"] = df["DESCRIPTION"].apply(extract_vehicle_count)

print("=== Category distribution ===")
print(df["category"].value_counts())
pct_unclassified = (df["category"] == "Other_unclassified").mean() * 100
print(f"\n% unclassified: {pct_unclassified:.1f}%")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to:\n  {OUTPUT_CSV}")
