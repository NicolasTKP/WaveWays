import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

ds = xr.open_dataset("data\\Bathymetry\\GEBCO_2024_sub_ice_topo.nc")
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


fig, ax = plt.subplots(figsize=(10, 8))
ocean_mask.plot(ax=ax, cmap='binary', add_colorbar=False)
malaysia_eez.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1.5)

# Set map limits
ax.set_xlim(99, 120)
ax.set_ylim(0, 8)

plt.title("Malaysia EEZ Overlaid on Bathymetry (Ocean Areas)")
plt.show()