import xarray as xr

url = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global"
ds = xr.open_dataset(url, decode_times=True, decode_timedelta=True)

point = ds.sel(
    longitude=104,
    latitude=2.5,  # nearest 0.5 grid to 2.8
    method="nearest"
)


latest = point["Tdir"].isel(time=-1)
print(latest.time.values, latest.values)
latest = point["Tper"].isel(time=-1)
print(latest.time.values, latest.values)
latest = point["Thgt"].isel(time=-1)
print(latest.time.values, latest.values)


latest = point["wdir"].isel(time=-1)
print(latest.time.values, latest.values)
latest = point["wper"].isel(time=-1)
print(latest.time.values, latest.values)
latest = point["whgt"].isel(time=-1)
print(latest.time.values, latest.values)