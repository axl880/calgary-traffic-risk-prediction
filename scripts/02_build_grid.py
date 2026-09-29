"""
Script 2/5 - Build the spatial grid and aggregate risk
Project: Calgary Traffic Collision Risk Prediction
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import os

# ============================================================
# FIXED PROJECT PATHS
# ============================================================
BASE_DIR = r"C:\Portafolio\calgary-traffic-risk"
INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "traffic_incidents_categorized.csv")
OUTPUT_GPKG = os.path.join(BASE_DIR, "data", "processed", "grid_risk_calgary.gpkg")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "grid_risk_calgary.csv")

CELL_SIZE = 500  # meters

# --- Input check ---
if not os.path.exists(INPUT_CSV):
    raise SystemExit(
        f"\nERROR: file not found at:\n  {INPUT_CSV}\n"
        f"Run 01_categorize.py first.\n"
    )

df = pd.read_csv(INPUT_CSV)
print(f"File loaded successfully: {INPUT_CSV}")
print(f"Total incidents: {len(df)}")
print(f"Columns found: {list(df.columns)}\n")

required_columns = ["category", "lane_blocked", "vehicle_count", "Longitude", "Latitude", "QUADRANT", "id"]
missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise SystemExit(f"\nERROR: missing columns {missing} in the input CSV.\n")

# --- Convert to GeoDataFrame and reproject to meters ---
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:4326"
)
gdf = gdf.to_crs("EPSG:26911")

minx, miny, maxx, maxy = gdf.total_bounds
print(f"Data extent (UTM): {minx:.0f}, {miny:.0f} to {maxx:.0f}, {maxy:.0f}")

# --- Build the fishnet ---
cells = []
x = minx
cell_id = 0
while x < maxx:
    y = miny
    while y < maxy:
        cells.append({"cell_id": cell_id, "geometry": box(x, y, x + CELL_SIZE, y + CELL_SIZE)})
        cell_id += 1
        y += CELL_SIZE
    x += CELL_SIZE

grid = gpd.GeoDataFrame(cells, crs="EPSG:26911")
print(f"Cells created: {len(grid)}")

# --- Spatial join ---
joined = gpd.sjoin(gdf, grid, how="left", predicate="within")

# --- Aggregate by cell ---
# NOTE: 'quadrant' is NOT computed here from the incidents -> that would
# cause data leakage (a cell with zero incidents would have no way to be
# assigned a quadrant, and that "no data" flag would end up almost
# perfectly correlated with the target). Quadrant is computed further
# below, purely from geometry, independent of whether incidents occurred.
agg = joined.groupby("cell_id").agg(
    total_incidents=("id", "count"),
    pct_lane_blocked=("lane_blocked", "mean"),
    pct_multi_vehicle_collision=("category", lambda s: (s == "Multi_vehicle_collision").mean()),
    pct_single_vehicle_collision=("category", lambda s: (s == "Single_vehicle_collision").mean()),
    pct_pedestrian_cyclist=("category", lambda s: s.isin(["Pedestrian", "Cyclist"]).mean()),
    avg_vehicle_count=("vehicle_count", "mean"),
).reset_index()

grid_final = grid.merge(agg, on="cell_id", how="left")
grid_final["total_incidents"] = grid_final["total_incidents"].fillna(0).astype(int)
for col in ["pct_lane_blocked", "pct_multi_vehicle_collision",
            "pct_single_vehicle_collision", "pct_pedestrian_cyclist", "avg_vehicle_count"]:
    grid_final[col] = grid_final[col].fillna(0)

# --- Quadrant computed geometrically (independent of incidents) ---
# Reference point: approximate crossing of Centre St. and the Bow/Elbow
# river, the historical point from which Calgary divides its NW/NE/SW/SE
# quadrants.
CENTER_LAT, CENTER_LON = 51.0447, -114.0719
centroids = grid_final.geometry.centroid.to_crs("EPSG:4326")
lat = centroids.y
lon = centroids.x
ns = np.where(lat >= CENTER_LAT, "N", "S")
ew = np.where(lon >= CENTER_LON, "E", "W")
grid_final["quadrant"] = [n + e for n, e in zip(ns, ew)]
print("\n=== Quadrant distribution (computed geometrically) ===")
print(grid_final["quadrant"].value_counts())

print(f"\nCells with at least 1 incident: {(grid_final['total_incidents'] > 0).sum()} of {len(grid_final)}")

# --- Target variable: risk level by quartiles ---
active = grid_final[grid_final["total_incidents"] > 0].copy()
active["risk_level"] = pd.qcut(active["total_incidents"], q=4, labels=["Low", "Medium", "High", "Very_high"])
grid_final = grid_final.merge(active[["cell_id", "risk_level"]], on="cell_id", how="left")
grid_final["risk_level"] = grid_final["risk_level"].cat.add_categories("No_data").fillna("No_data")

print("\n=== Risk level distribution ===")
print(grid_final["risk_level"].value_counts())

# --- Save ---
os.makedirs(os.path.dirname(OUTPUT_GPKG), exist_ok=True)
grid_final.to_file(OUTPUT_GPKG, driver="GPKG")
grid_final.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved:")
print(f"  {OUTPUT_GPKG}")
print(f"  {OUTPUT_CSV}")
