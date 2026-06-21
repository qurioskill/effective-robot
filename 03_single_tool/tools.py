"""Tools for Example 3 — a single weather tool backed by Open-Meteo.

Open-Meteo is free and requires no API key, which makes it perfect for a live
demo. We use two of its endpoints inside one tool:

* Geocoding  : turn a city name into latitude/longitude.
* Forecast   : turn lat/lon into the current weather.

A LangChain "tool" is just a Python function decorated with ``@tool``. The
docstring and type hints become the schema the model sees, so write them
clearly — the model uses them to decide when and how to call the tool.
"""

from __future__ import annotations

import requests
from langchain_core.tools import tool

# A short, friendly map from Open-Meteo's numeric WMO weather codes to text.
# (Trimmed to the common ones to keep the example readable.)
_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    95: "thunderstorm",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city, e.g. "Toronto" or "Paris".

    Returns:
        A short human-readable description of the current weather, or an error
        message if the city could not be found or the API call failed.
    """
    # 1) Geocode the city name -> latitude/longitude.
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
    except requests.RequestException as exc:
        return f"Could not reach the geocoding service: {exc}"

    if not results:
        return f"Sorry, I couldn't find a city called '{city}'."

    place = results[0]
    lat, lon = place["latitude"], place["longitude"]
    label = f"{place['name']}, {place.get('country', '')}".strip(", ")

    # 2) Fetch the current weather for those coordinates.
    try:
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=10,
        )
        forecast.raise_for_status()
        current = forecast.json().get("current_weather", {})
    except requests.RequestException as exc:
        return f"Could not reach the weather service: {exc}"

    if not current:
        return f"No current weather data available for {label}."

    temp = current.get("temperature")
    wind = current.get("windspeed")
    code = current.get("weathercode")
    description = _WEATHER_CODES.get(code, "unknown conditions")

    return (
        f"Current weather in {label}: {description}, "
        f"{temp}°C, wind {wind} km/h."
    )
