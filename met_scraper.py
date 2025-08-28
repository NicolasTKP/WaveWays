# met_scraper.py
import requests, re
from bs4 import BeautifulSoup

URL = "https://www.met.gov.my/en/forecast/marine/shipping/"

def fetch_met_forecast():
    try:
        r = requests.get(URL, timeout=15)
        r.raise_for_status()
        html = r.text
        open("met_page_debug.html", "w", encoding="utf-8").write(html)
    except Exception as e:
        print(f"[met_scraper] ERROR: failed to fetch page: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    forecasts = []
    # Find marine forecast table(s)
    tables = soup.find_all("table")
    for tbl in tables:
        rows = tbl.find_all("tr")
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cols) < 3: 
                continue

            region = cols[0]
            wave_text = cols[1]
            wind_text = cols[2]

            # Extract numeric values
            wave_match = re.findall(r"(\d+(\.\d+)?)", wave_text)
            wind_match = re.findall(r"(\d+)", wind_text)

            wave_val = float(wave_match[-1][0]) if wave_match else 0.0
            wind_val = float(wind_match[-1]) if wind_match else 0.0

            forecasts.append({
                "region": region,
                "wave": wave_val,
                "wind": wind_val
            })

    print(f"[met_scraper] Parsed {len(forecasts)} region forecasts")
    return forecasts

if __name__ == "__main__":
    print(fetch_met_forecast())

# save from metscraper
