import geopandas as gpd
import matplotlib.pyplot as plt

# Load the shapefile
eez = gpd.read_file("data\\Fishing_Zones\\EEZ_land_union_v4_202410.shp")
malaysia_eez = eez[eez['TERRITORY1'] == 'Malaysia']

# Load downloaded world shapefile
world = gpd.read_file("data\\World_Shape\\ne_110m_admin_0_countries.shp")

fig, ax = plt.subplots(figsize=(10, 6))
world.plot(ax=ax, color='lightgrey', edgecolor='black')
malaysia_eez.plot(ax=ax, color='steelblue')
ax.set_xlim(90, 130)
ax.set_ylim(-10, 15)
ax.set_title("Malaysia EEZ with World Basemap")
plt.show()