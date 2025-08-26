import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


locations = {
    # Practical, coverage-first split of the Malacca Strait at 3.00°N.
    # Northern section spans ~3.00°N–5.95°N and across the full width of the strait.
    "Northern_Straits_of_Malacca": [
        {"lat": 5.9500, "lon": 98.0250},  # near I-M-T tripoint sector (NW entrance line vicinity)
        {"lat": 5.9500, "lon": 100.9000}, # off Kedah/Penang side (NE)
        {"lat": 3.0000, "lon": 101.4000}, # split line at 3°N (SE)
        {"lat": 3.0000, "lon": 99.0000}   # split line at 3°N (SW)
    ],

    # Southern section uses the IHO eastern exit line points and reaches up to the 3.00°N split.
    "Southern_Straits_of_Malacca": [
        {"lat": 3.0000, "lon": 100.8000},  # split line near Sumatra side (NW)
        {"lat": 3.0000, "lon": 103.3000},  # split line near Malay Peninsula side (NE)
        {"lat": 1.2667, "lon": 103.5167},  # Tanjong Piai (IHO eastern limit point)
        {"lat": 1.1667, "lon": 103.3917},  # Klein Karimoen (IHO eastern limit point)
        {"lat": 1.1000, "lon": 102.9667}   # Tanjong Kedabu on Sumatra coast (IHO southern edge)
    ],

    # Islands below use coverage rectangles that fully contain the island;
    # replace with detailed coastlines if you later load shore polygons.
    "Tioman_Island": [
        {"lat": 2.9000, "lon": 104.0500},  # NW
        {"lat": 2.9000, "lon": 104.3000},  # NE
        {"lat": 2.7000, "lon": 104.3000},  # SE
        {"lat": 2.7000, "lon": 104.0500}   # SW
    ],
    "Bunguran_Island": [  # aka Natuna Besar
        {"lat": 4.2000, "lon": 107.8000},  # NW
        {"lat": 4.2000, "lon": 108.6000},  # NE
        {"lat": 3.6000, "lon": 108.6000},  # SE
        {"lat": 3.6000, "lon": 107.8000}   # SW
    ],
    "Layang_Layang_Island": [
        {"lat": 7.3900, "lon": 113.8200},  # NW
        {"lat": 7.3900, "lon": 113.8700},  # NE
        {"lat": 7.3500, "lon": 113.8700},  # SE
        {"lat": 7.3500, "lon": 113.8200}   # SW
    ],
    "Labuan": [
        {"lat": 5.3600, "lon": 115.1600},  # NW
        {"lat": 5.3600, "lon": 115.3600},  # NE
        {"lat": 5.2200, "lon": 115.3600},  # SE
        {"lat": 5.2200, "lon": 115.1600}   # SW
    ]
}

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

# Plot the locations from the dictionary
for location_name, points in locations.items():
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    ax.scatter(lons, lats, s=50, label=location_name, marker='x')
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        ax.text(lon, lat, f"{location_name} Point {i+1}", fontsize=7, ha='left', va='bottom')

# Set map limits
ax.set_xlim(99, 120)
ax.set_ylim(0, 8)

plt.title("Malaysia EEZ Overlaid on Bathymetry (Ocean Areas) with Key Locations")
plt.legend()
plt.show()
