import xarray as xr

ds_bathymetry = xr.open_dataset("data/Bathymetry/GEBCO_2025_sub_ice.nc")
print(ds_bathymetry['lat'])