"""
Script 5/5 - Generate predictions and risk map
Project: Calgary Traffic Collision Risk Prediction

Applies the trained model to ALL cells (not just the test set) and
generates:
  1. A GeoPackage to open in ArcGIS Pro / QGIS
  2. An interactive web map (HTML) for your portfolio / GitHub
"""

import pandas as pd
import geopandas as gpd
import joblib
import folium
import os

BASE_DIR = r"C:\Portafolio\calgary-traffic-risk"
FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "grid_features_final.csv")
GRID_GEOM_GPKG = os.path.join(BASE_DIR, "data", "processed", "grid_risk_calgary.gpkg")
MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "random_forest_model.joblib")

OUTPUT_GPKG = os.path.join(BASE_DIR, "data", "processed", "final_risk_map.gpkg")
OUTPUT_HTML = os.path.join(BASE_DIR, "data", "processed", "calgary_risk_map.html")

# --- Load model and features ---
package = joblib.load(MODEL_PATH)
model = package["model"]
feature_cols = package["features"]
print(f"Model loaded. Expected features: {feature_cols}")

# --- Load feature table and geometry separately, then merge ---
df = pd.read_csv(FEATURES_CSV)
grid_geom = gpd.read_file(GRID_GEOM_GPKG)[["cell_id", "geometry"]]

# --- Rebuild the same dummy columns used during training ---
X_full = df[["total_road_length_m", "pct_arterial_road", "num_road_segments", "road_density"]].copy()
X_full = X_full.join(pd.get_dummies(df["quadrant"], prefix="quad", drop_first=True))

# Make sure the columns match training exactly
# (if a quadrant category is missing in the current data, add it as 0)
for col in feature_cols:
    if col not in X_full.columns:
        X_full[col] = 0
X_full = X_full[feature_cols]

# --- Predict high-risk probability for ALL cells ---
df["high_risk_probability"] = model.predict_proba(X_full)[:, 1]
df["high_risk_prediction"] = model.predict(X_full)

print("\n=== Predicted probability distribution ===")
print(df["high_risk_probability"].describe())

# --- Merge with geometry and reproject to WGS84 for web maps ---
map_gdf = grid_geom.merge(
    df[["cell_id", "high_risk_probability", "high_risk_prediction", "risk_level",
        "risk_level_normalized", "total_incidents", "quadrant"]],
    on="cell_id", how="left"
)
map_gdf = map_gdf.to_crs("EPSG:4326")

# --- Save GeoPackage for ArcGIS Pro / QGIS ---
if os.path.exists(OUTPUT_GPKG):
    os.remove(OUTPUT_GPKG)
map_gdf.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"\nSaved for ArcGIS Pro/QGIS: {OUTPUT_GPKG}")
print("Open it there and symbolize 'high_risk_probability' with graduated colors.")

# --- Build the interactive folium map ---
calgary_center = [51.0447, -114.0719]
m = folium.Map(location=calgary_center, zoom_start=11, tiles="cartodbpositron")

# Only draw cells with real road activity (avoids painting the whole map gray)
map_gdf_active = map_gdf[map_gdf["high_risk_probability"].notna()].copy()

folium.Choropleth(
    geo_data=map_gdf_active.__geo_interface__,
    data=map_gdf_active,
    columns=["cell_id", "high_risk_probability"],
    key_on="feature.properties.cell_id",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.1,
    legend_name="Predicted probability of high collision risk",
).add_to(m)

# Tooltip with detail on hover
folium.GeoJson(
    map_gdf_active,
    style_function=lambda x: {"fillOpacity": 0, "weight": 0},
    tooltip=folium.GeoJsonTooltip(
        fields=["quadrant", "total_incidents", "high_risk_probability"],
        aliases=["Quadrant:", "Historical incidents:", "High-risk probability:"],
        localize=True,
    ),
).add_to(m)

m.save(OUTPUT_HTML)
print(f"\nInteractive map saved: {OUTPUT_HTML}")
print("Open it in your browser to review it, and upload it with your project to GitHub.")
