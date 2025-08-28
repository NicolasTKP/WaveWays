# Astar.py
import heapq
from math import sqrt
from met_scraper import fetch_met_forecast

# -----------------------------
# Helpers
# -----------------------------
def haversine(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_neighbors(node, step=0.5):
    lat, lon = node
    return [
        (lat+step, lon),
        (lat-step, lon),
        (lat, lon+step),
        (lat, lon-step),
        (lat+step, lon+step),
        (lat-step, lon-step),
        (lat+step, lon-step),
        (lat-step, lon+step),
    ]

# -----------------------------
# Weather Blocking
# -----------------------------
def is_blocked(point, forecasts, max_wave=3.0, max_wind=20.0):
    """
    Dummy mapping: currently all regions treated same.
    Extend this by mapping 'region' polygons (manual bounding boxes).
    """
    for f in forecasts:
        if f["wave"] > max_wave or f["wind"] > max_wind:
            # For now block everything if condition exceeded
            return True
    return False

# -----------------------------
# A* Search
# -----------------------------
def astar(start, goal, forecasts, step=0.5, max_wave=3.0, max_wind=20.0):
    print(f"[A*] Start={start} Goal={goal} wave≤{max_wave}m wind≤{max_wind}kt step={step}°")

    min_lat = min(start[0], goal[0]) - 2
    max_lat = max(start[0], goal[0]) + 2
    min_lon = min(start[1], goal[1]) - 2
    max_lon = max(start[1], goal[1]) + 2
    print(f"[A*] ROI lat {min_lat}..{max_lat} lon {min_lon}..{max_lon}")

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    gscore = {start: 0}
    fscore = {start: haversine(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if haversine(current, goal) < step:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for n in get_neighbors(current, step):
            if not (min_lat <= n[0] <= max_lat and min_lon <= n[1] <= max_lon):
                continue
            if is_blocked(n, forecasts, max_wave, max_wind):
                continue

            tentative = gscore[current] + haversine(current, n)
            if n not in gscore or tentative < gscore[n]:
                came_from[n] = current
                gscore[n] = tentative
                fscore[n] = tentative + haversine(n, goal)
                heapq.heappush(open_set, (fscore[n], n))

    return None

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    start = (1.3, 103.8)   # Singapore
    goal = (22.3, 114.2)   # Hong Kong

    forecasts = fetch_met_forecast()
    if not forecasts:
        print("[A*] Warning: no forecast polygons parsed; no weather blocks will be applied.")

    path = astar(start, goal, forecasts, step=0.5, max_wave=3.0, max_wind=20.0)
    if path:
        print(f"[A*] Path found with {len(path)} nodes")
        for p in path:
            print(p)
    else:
        print("[A*] No path found")

#Save path to json
import json
with open("astar_path.json", "w") as f:
    json.dump(path, f)

with open("met_regions.json", "w") as f:
    json.dump(forecasts, f)
