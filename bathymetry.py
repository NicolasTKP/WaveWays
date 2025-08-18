import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ds = xr.open_dataset("data\\Bathymetry\\GEBCO_2025_sub_ice.nc")
print(ds)


# Subset to Malaysia region: Longitude 99–120, Latitude 0 to 8
ds_subset = ds.sel(
    lon=slice(99, 120),
    lat=slice(0, 8)
)

# Load the ocean mask based on elevation
elevation = ds_subset['elevation']
ocean_mask = xr.where(elevation < 0, 1, np.nan)

# Load and query EEZ shapefile
eez = gpd.read_file("data\\Fishing_Zones\\EEZ_land_union_v4_202410.shp")
malaysia_eez = eez[eez['TERRITORY1'] == 'Malaysia']

# load ports data
ports_df = pd.read_excel("data\\Ports\\ports.xlsx")
ports_df[["lat", "lon"]] = ports_df["Decimal"].str.split(",", expand=True).astype(float)
ports_gdf = gpd.GeoDataFrame(
    ports_df,
    geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(10, 8))
ocean_mask.plot(ax=ax, cmap='binary', add_colorbar=False)
malaysia_eez.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1.5)
ports_gdf.plot(ax=ax, color='blue', markersize=30, label="Ports")

# optional: ports label
for x, y, label in zip(ports_gdf.geometry.x, ports_gdf.geometry.y, ports_gdf["Ports"]):
    ax.text(x, y, label, fontsize=8, ha='right')

# Set map limits
ax.set_xlim(99, 120)
ax.set_ylim(0, 8)

plt.title("Malaysia EEZ Overlaid on Bathymetry (Ocean Areas)")
plt.legend()
plt.show()