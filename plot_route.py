# plot_route.py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import json

def plot_route(path, regions, outfile="route_map.png"):
    # Setup map
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([100, 120, -5, 25], crs=ccrs.PlateCarree())

    # Add coastlines & land
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.gridlines(draw_labels=True)

    # Plot route
    lats, lons = zip(*path)
    ax.plot(lons, lats, "r-o", markersize=3, linewidth=1.5, label="Route")

    # Plot MET polygons (if polygon data exists)
    for region in regions:
        # Check if polygon data exists in the region
        if "polygon" in region and region["polygon"]:
            poly = region["polygon"]
            xs, ys = zip(*poly)
            ax.plot(ys, xs, "b--", alpha=0.7)  # outline
            ax.fill(ys, xs, color="blue", alpha=0.2)  # fill
        else:
            # Skip regions without polygon data
            print(f"Skipping region without polygon data: {region.get('region', 'Unknown')}")

    plt.legend()
    plt.title("Marine Route with MET Weather Polygons")
    plt.savefig(outfile, dpi=200)
    plt.show()

if __name__ == "__main__":
    # Example: load from JSON (you can dump these after running A*)
    with open("astar_path.json") as f:
        path = json.load(f)

    with open("met_regions.json") as f:
        regions = json.load(f)

    plot_route(path, regions)
