import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-dashboard")


@mcp.tool()
def get_current_weather(location: str) -> str:
    """Get the current real-time weather for a city or location.

    Args:
        location: City or place name, e.g. "Tokyo", "Bengaluru", "New York".
    """
    url = f"https://wttr.in/{requests.utils.quote(location)}"
    try:
        resp = requests.get(
            url,
            params={"format": "j1"},
            headers={"User-Agent": "curl/8.0"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        place = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        today = data["weather"][0]

        return (
            f"Location: {place}, {country}\n"
            f"Condition: {current['weatherDesc'][0]['value']}\n"
            f"Temperature: {current['temp_C']}°C (feels like {current['FeelsLikeC']}°C)\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind: {current['windspeedKmph']} km/h {current['winddir16Point']}\n"
            f"Visibility: {current['visibility']} km\n"
            f"UV index: {current['uvIndex']}\n"
            f"Today's range: {today['mintempC']}°C to {today['maxtempC']}°C"
        )
    except requests.exceptions.RequestException as e:
        return f"Could not fetch weather for '{location}': {e}"
    except (KeyError, IndexError, ValueError):
        return f"Could not understand the weather data for '{location}'. Check the location name."


if __name__ == "__main__":
    mcp.run(transport="stdio")