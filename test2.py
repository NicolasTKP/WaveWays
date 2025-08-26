import requests
from bs4 import BeautifulSoup
import re

def scrape_weather_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    weather_data = {}

    # Find the main content area where the weather information is displayed
    # Based on a quick inspection of the page, the data seems to be within tables or specific divs.
    # I'll look for common patterns like 'table', 'div', 'p' with relevant text.

    # Example: Extracting data from tables. This part needs to be adapted based on the actual HTML structure.
    # Let's assume the weather data is in a table with class 'weather-table' or similar.
    # This is a placeholder and will need refinement after inspecting the actual HTML.

    # The page structure is complex, with multiple tables and divs.
    # I'll try to find the main content area and then iterate through its children.
    
    # A common pattern for content is often within a div with a specific ID or class.
    # Let's try to find all tables and then filter them.
    tables = soup.find_all('table')
    
    if not tables:
        print("No tables found on the page.")
        return weather_data

    # Assuming the relevant table is the first one or has a specific structure
    # This part requires manual inspection of the HTML to be precise.
    # For now, I'll iterate through all tables and try to extract data.
    
    for i, table in enumerate(tables):
        table_data = []
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = table.find_all('tr')
        
        # Skip the first row if headers were found, as it's likely the header row itself
        start_row_index = 1 if headers else 0
        
        for row in rows[start_row_index:]:
            cols = row.find_all('td') # Only look for <td> for data rows
            cols = [ele.get_text(strip=True) for ele in cols]
            row_data = [ele for ele in cols if ele]
            
            if headers and len(row_data) == len(headers):
                row_dict = dict(zip(headers, row_data))
                
                # Further parse the 'Forecast' string if it exists
                if 'Forecast' in row_dict:
                    forecast_str = row_dict['Forecast']
                    parsed_forecast = {}
                    
                    # Split by "Morning:", "Afternoon:", "Night:"
                    # Use regex to split the forecast string into periods
                    period_matches = re.findall(r'(Morning:|Afternoon:|Night:)(.*?)(?=Morning:|Afternoon:|Night:|$)', forecast_str)
                    
                    for period_tag, period_content in period_matches:
                        period_name = period_tag.replace(':', '').strip()
                        
                        # Extract Weather description
                        # The weather description is everything before "Wind Direction:"
                        weather_desc_match = re.match(r'(.+?)(?:Wind Direction:|$)', period_content)
                        weather_desc = weather_desc_match.group(1).strip() if weather_desc_match else period_content.strip()

                        forecast_details = {
                            'Weather': weather_desc if weather_desc else None
                        }
                        parsed_forecast[period_name] = forecast_details
                    
                    row_dict['Forecast'] = parsed_forecast
                table_data.append(row_dict)
            else:
                table_data.append(row_data) # Fallback if headers don't match row length
        
        if table_data:
            weather_data[f"table_{i+1}"] = table_data

    return weather_data

if __name__ == "__main__":
    locations = {
        "Northern_Straits_of_Malacca": "Sh002",
        "Southern_Straits_of_Malacca": "Sh003",
        "Tioman_Island": "Sh005",
        "Bunguran_Island": "Sh007",
        "Layang_Layang_Island": "Sh010",
        "Labuan": "Sh012"
    }

    all_scraped_data = {}

    for location_name, location_code in locations.items():
        url = f"https://www.met.gov.my/en/forecast/marine/shipping/{location_code}/"
        print(f"\nScraping data for {location_name} ({location_code})...")
        scraped_data = scrape_weather_data(url)
        
        if scraped_data:
            all_scraped_data[location_name] = scraped_data
            print(f"Successfully scraped data for {location_name}.")
        else:
            print(f"Failed to scrape data for {location_name}.")

    if all_scraped_data:
        print("\n--- All Scraped Weather Data ---")
        for location, data in all_scraped_data.items():
            print(f"\nLocation: {location}")
            for key, value in data.items():
                print(f"  {key}: {value}")
    else:
        print("\nNo data was scraped for any location.")
