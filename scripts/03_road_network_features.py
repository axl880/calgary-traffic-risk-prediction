"""
Script 3/5 - Road network features per cell
Computes, for each grid cell: total road length, arterial road length,
and segment count (a proxy for complexity/crossings).
Then merges them with the risk table built previously.
"""

import geopandas as gpd
import pandas as pd
import os

BASE_DIR = r"C:\Portafolio\calgary-traffic-risk"
ROADS_PATH = os.path.join(BASE_DIR, "data", "raw", "street_centreline.geojson")
GRID_GPKG = os.path.join(BASE_DIR, "data", "processed", "grid_risk_calgary.gpkg")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "grid_features_final.csv")
OUTPUT_GPKG = os.path.join(BASE_DIR, "data", "processed", "grid_features_final.gpkg")

CELL_SIZE = 500  # must match the value used when building the grid

# --- Checks ---
for path in [ROADS_PATH, GRID_GPKG]:
    if not os.path.exists(path):
        raise SystemExit(f"\nERROR: file not found at:\n  {path}\n")

# --- Load data ---
roads = gpd.read_file(ROADS_PATH)
grid = gpd.read_file(GRID_GPKG)
print(f"Road segments: {len(roads)}")
print(f"Grid cells: {len(grid)}")

# --- Align coordinate systems (meters) ---
roads = roads.to_crs("EPSG:26911")
if grid.crs is None or grid.crs.to_epsg() != 26911:
    grid = grid.to_crs("EPSG:26911")

# --- Classify arterial roads based on ctp_class ---
ARTERIAL_CLASSES = ["Arterial Street", "Industrial Arterial", "Local Arterial", "Skeletal Road"]
roads["is_arterial"] = roads["ctp_class"].isin(ARTERIAL_CLASSES)
print(f"\nArterial segments: {roads['is_arterial'].sum()} of {len(roads)}")

# --- Clip road segments by cell ---
# This is more precise than a simple spatial join: if a road crosses a
# cell boundary, only the portion that is actually inside is counted.
print("\nClipping segments by cell (this can take a couple of minutes)...")
roads_slim = roads[["ctp_class", "is_arterial", "geometry"]]
grid_slim = grid[["cell_id", "geometry"]]
clipped = gpd.overlay(roads_slim, grid_slim, how="intersection")

clipped["length_m"] = clipped.geometry.length
clipped["arterial_length_m"] = clipped["length_m"].where(clipped["is_arterial"], 0)

# --- Aggregate by cell ---
road_agg = clipped.groupby("cell_id").agg(
    total_road_length_m=("length_m", "sum"),
    arterial_road_length_m=("arterial_length_m", "sum"),
    num_road_segments=("length_m", "count"),
).reset_index()

road_agg["pct_arterial_road"] = (
    road_agg["arterial_road_length_m"] / road_agg["total_road_length_m"]
).fillna(0)

cell_area_m2 = CELL_SIZE ** 2
road_agg["road_density"] = road_agg["total_road_length_m"] / cell_area_m2

# --- Merge with the existing risk grid ---
grid_with_risk = pd.read_csv(
    os.path.join(BASE_DIR, "data", "processed", "grid_risk_calgary.csv")
)

final = grid_with_risk.merge(road_agg, on="cell_id", how="left")
for col in ["total_road_length_m", "arterial_road_length_m", "num_road_segments",
            "pct_arterial_road", "road_density"]:
    final[col] = final[col].fillna(0)

print("\n=== New feature statistics ===")
print(final[["total_road_length_m", "pct_arterial_road", "num_road_segments", "road_density"]].describe())

# --- Quick sanity check: Very_high risk cells should have, on average,
# more road length and more segments than Low risk cells ---
print("\n=== Average road features by risk level ===")
print(final.groupby("risk_level")[["total_road_length_m", "pct_arterial_road", "num_road_segments"]].mean())

# --- NORMALIZED risk variable: incidents per km of road ---
# Minimum threshold of 200m: below that, the rate becomes unstable
# (a single incident in a cell with very little road inflates the rate
# artificially). Those cells are flagged separately, not discarded.
MIN_THRESHOLD_M = 200

final["incidents_per_km"] = None
valid_mask = final["total_road_length_m"] >= MIN_THRESHOLD_M
final.loc[valid_mask, "incidents_per_km"] = (
    final.loc[valid_mask, "total_incidents"]
    / (final.loc[valid_mask, "total_road_length_m"] / 1000)
)

valid_rows = final[valid_mask & (final["total_incidents"] > 0)].copy()
valid_rows["risk_level_normalized"] = pd.qcut(
    valid_rows["incidents_per_km"], q=4, labels=["Low", "Medium", "High", "Very_high"]
)
final = final.merge(valid_rows[["cell_id", "risk_level_normalized"]], on="cell_id", how="left")
final["risk_level_normalized"] = final["risk_level_normalized"].cat.add_categories(
    "No_data_or_insufficient"
).fillna("No_data_or_insufficient")

print(f"\nCells excluded from the rate calculation for having less than {MIN_THRESHOLD_M}m of road: "
      f"{(~valid_mask).sum()} of {len(final)}")
print("\n=== risk_level_normalized distribution (rate per km) ===")
print(final["risk_level_normalized"].value_counts())

# --- Comparison: how many cells change category between the two approaches ---
comparable = final[
    (final["risk_level"] != "No_data")
    & (final["risk_level_normalized"] != "No_data_or_insufficient")
]
match_pct = (comparable["risk_level"] == comparable["risk_level_normalized"]).mean() * 100
print(f"\n% of cells where raw risk and normalized risk agree on the same category: {match_pct:.1f}%")
print("(a low number here is interesting: it means the ranking changes a lot")
print(" depending on whether you measure absolute volume or rate per road exposure")
print(" - worth mentioning in the README)")

final.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print("This is the final table used to train the model (now with road network features).")
