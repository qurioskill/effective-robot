"""Tools for Example 4 — several small tools the agent can chain together.

We deliberately split "find the city's coordinates" and "get the weather for
coordinates" into two tools. That forces the agent to *chain* calls:

    geocode_city("Berlin") -> (lat, lon) -> get_weather(lat, lon)

Chaining is the whole point of this example: the model figures out the order on
its own.

Tools:
    * geocode_city      - city name        -> latitude/longitude
    * get_weather       - latitude/longitude -> current weather
    * get_current_time  - IANA timezone     -> current local time
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from langchain_core.tools import tool

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
def geocode_city(city: str) -> str:
    """Look up the latitude and longitude of a city by name.

    Args:
        city: The city name, e.g. "Berlin".

    Returns:
        A string like "Berlin, Germany -> lat=52.52, lon=13.41", or an error
        message if the city is not found.
    """
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
    except requests.RequestException as exc:
        return f"Could not reach the geocoding service: {exc}"

    if not results:
        return f"No city found named '{city}'."

    place = results[0]
    label = f"{place['name']}, {place.get('country', '')}".strip(", ")
    return f"{label} -> lat={place['latitude']}, lon={place['longitude']}"


@tool
def get_weather(latitude: float, longitude: float) -> str:
    """Get the current weather for a latitude/longitude pair.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.

    Returns:
        A short description of the current weather, or an error message.
    """
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
        current = resp.json().get("current_weather", {})
    except requests.RequestException as exc:
        return f"Could not reach the weather service: {exc}"

    if not current:
        return "No current weather data available for those coordinates."

    description = _WEATHER_CODES.get(current.get("weathercode"), "unknown conditions")
    return (
        f"{description}, {current.get('temperature')}°C, "
        f"wind {current.get('windspeed')} km/h."
    )


@tool
def get_current_time(timezone: str) -> str:
    """Get the current local time in a given IANA timezone.

    Args:
        timezone: An IANA timezone name, e.g. "Europe/Berlin" or
            "America/Toronto".

    Returns:
        The current local date and time, or an error message for an unknown
        timezone.
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        return f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/Berlin'."

    return now.strftime("%Y-%m-%d %H:%M:%S %Z")
