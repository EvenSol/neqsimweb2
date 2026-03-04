"""
Weather integration — fetch real ambient conditions and assess impact on process.

Uses Open-Meteo API (free, no API key required) to pull current & forecast
weather for any location.  Results include:
  • Current conditions (temperature, humidity, wind)
  • 7-day hourly forecast summary (min/max/avg temperature)
  • Design-basis comparison (actual vs. design ambient temperature)
  • Cooler capacity warnings & derating estimates
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CurrentWeather:
    """Snapshot of current conditions at the site."""
    temperature_C: float
    relative_humidity_pct: float
    wind_speed_m_s: float
    wind_direction_deg: float
    apparent_temperature_C: float
    cloud_cover_pct: float
    precipitation_mm: float
    timestamp: str  # ISO-8601


@dataclass
class ForecastDay:
    """Aggregated daily forecast."""
    date: str
    temp_min_C: float
    temp_max_C: float
    temp_avg_C: float
    humidity_avg_pct: float
    wind_max_m_s: float
    precipitation_sum_mm: float


@dataclass
class CoolerImpact:
    """Assessment of weather impact on air-cooled equipment."""
    design_ambient_C: float
    current_ambient_C: float
    delta_C: float
    capacity_factor: float  # 1.0 = full capacity; <1 = derated
    status: str  # "OK", "WARNING", "CRITICAL"
    message: str


@dataclass
class WeatherResult:
    """Top-level result returned to the chat layer."""
    location_name: str
    latitude: float
    longitude: float
    current: CurrentWeather
    forecast: List[ForecastDay]
    cooler_impact: Optional[CoolerImpact]
    recommendations: List[str]
    raw_url: str  # API URL for traceability


# ---------------------------------------------------------------------------
# Open-Meteo API fetch
# ---------------------------------------------------------------------------

_OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude={lat}&longitude={lon}"
    "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
    "precipitation,cloud_cover,wind_speed_10m,wind_direction_10m"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    "&forecast_days=7&timezone=auto"
)


def _fetch_open_meteo(lat: float, lon: float) -> dict:
    """Fetch weather data from Open-Meteo (free, no key)."""
    url = _OPEN_METEO_URL.format(lat=lat, lon=lon)
    req = urllib.request.Request(url, headers={"User-Agent": "NeqSimWeb/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        data["_request_url"] = url
        return data
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Weather API request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Geocoding helper (Open-Meteo geocoding API)
# ---------------------------------------------------------------------------

_GEOCODE_URL = (
    "https://geocoding-api.open-meteo.com/v1/search?name={name}&count=1&language=en&format=json"
)


def geocode_city(city_name: str) -> tuple[float, float, str]:
    """Return (lat, lon, display_name) for a city name."""
    url = _GEOCODE_URL.format(name=urllib.request.quote(city_name))
    req = urllib.request.Request(url, headers={"User-Agent": "NeqSimWeb/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Geocoding request failed: {exc}") from exc

    results = data.get("results")
    if not results:
        raise ValueError(f"Could not find location: {city_name}")
    r = results[0]
    display = f"{r.get('name', city_name)}, {r.get('country', '')}"
    return r["latitude"], r["longitude"], display


# ---------------------------------------------------------------------------
# Aggregate hourly→daily
# ---------------------------------------------------------------------------

def _aggregate_daily(hourly: dict) -> List[ForecastDay]:
    """Aggregate hourly arrays into per-day summaries."""
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    humids = hourly.get("relative_humidity_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    precips = hourly.get("precipitation", [])

    from collections import defaultdict
    days: Dict[str, dict] = defaultdict(lambda: {
        "temps": [], "humids": [], "winds": [], "precips": [],
    })
    for i, t in enumerate(times):
        date = t[:10]  # YYYY-MM-DD
        if i < len(temps):
            days[date]["temps"].append(temps[i])
        if i < len(humids):
            days[date]["humids"].append(humids[i])
        if i < len(winds):
            days[date]["winds"].append(winds[i])
        if i < len(precips):
            days[date]["precips"].append(precips[i])

    result = []
    for date in sorted(days.keys()):
        d = days[date]
        t_list = d["temps"]
        if not t_list:
            continue
        result.append(ForecastDay(
            date=date,
            temp_min_C=min(t_list),
            temp_max_C=max(t_list),
            temp_avg_C=sum(t_list) / len(t_list),
            humidity_avg_pct=sum(d["humids"]) / max(len(d["humids"]), 1),
            wind_max_m_s=max(d["winds"]) if d["winds"] else 0.0,
            precipitation_sum_mm=sum(d["precips"]),
        ))
    return result


# ---------------------------------------------------------------------------
# Cooler impact assessment
# ---------------------------------------------------------------------------

def _assess_cooler_impact(
    current_temp_C: float,
    design_ambient_C: float,
) -> CoolerImpact:
    """Estimate air-cooler capacity derating from ambient temperature."""
    delta = current_temp_C - design_ambient_C

    # Simple linear derating: ~2% capacity loss per °C above design
    if delta <= 0:
        factor = 1.0
        status = "OK"
        msg = (f"Ambient {current_temp_C:.1f}°C is at or below design basis "
               f"({design_ambient_C:.1f}°C). Coolers operating at full capacity.")
    elif delta <= 5:
        factor = max(1.0 - 0.02 * delta, 0.5)
        status = "WARNING"
        msg = (f"Ambient {current_temp_C:.1f}°C is {delta:.1f}°C above design "
               f"({design_ambient_C:.1f}°C). Estimated cooler capacity ~{factor*100:.0f}%.")
    else:
        factor = max(1.0 - 0.02 * delta, 0.5)
        status = "CRITICAL"
        msg = (f"Ambient {current_temp_C:.1f}°C is {delta:.1f}°C above design "
               f"({design_ambient_C:.1f}°C). Cooler derated to ~{factor*100:.0f}%. "
               f"Consider reducing throughput or supplemental cooling.")

    return CoolerImpact(
        design_ambient_C=design_ambient_C,
        current_ambient_C=current_temp_C,
        delta_C=round(delta, 1),
        capacity_factor=round(factor, 3),
        status=status,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_weather_analysis(
    model: Any = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    city: Optional[str] = None,
    design_ambient_C: float = 30.0,
) -> WeatherResult:
    """Fetch weather and assess process impact.

    Parameters
    ----------
    model : NeqSimProcessModel, optional
        Process model (used to list coolers for impact assessment).
    latitude, longitude : float, optional
        GPS coordinates.  If omitted, *city* is geocoded.
    city : str, optional
        City / location name (used if lat/lon not given).
    design_ambient_C : float
        Design-basis ambient temperature for air coolers.
    """
    # Resolve location
    location_name = city or "Unknown"
    if latitude is None or longitude is None:
        if city:
            latitude, longitude, location_name = geocode_city(city)
        else:
            raise ValueError("Provide latitude/longitude or a city name.")

    # Fetch weather
    raw = _fetch_open_meteo(latitude, longitude)
    raw_url = raw.pop("_request_url", "")

    # Parse current conditions
    cur = raw.get("current", {})
    current = CurrentWeather(
        temperature_C=cur.get("temperature_2m", 0.0),
        relative_humidity_pct=cur.get("relative_humidity_2m", 0.0),
        wind_speed_m_s=cur.get("wind_speed_10m", 0.0),
        wind_direction_deg=cur.get("wind_direction_10m", 0.0),
        apparent_temperature_C=cur.get("apparent_temperature", 0.0),
        cloud_cover_pct=cur.get("cloud_cover", 0.0),
        precipitation_mm=cur.get("precipitation", 0.0),
        timestamp=cur.get("time", ""),
    )

    # Aggregate forecast
    forecast = _aggregate_daily(raw.get("hourly", {}))

    # Cooler impact
    cooler = _assess_cooler_impact(current.temperature_C, design_ambient_C)

    # Recommendations
    recs: List[str] = []
    if cooler.status == "CRITICAL":
        recs.append("Consider reducing plant throughput until ambient temperature drops.")
        recs.append("Evaluate supplemental spray-cooling on fin-fan coolers.")
    elif cooler.status == "WARNING":
        recs.append("Monitor cooler outlet temperatures closely.")

    # Check for upcoming hot days
    hot_days = [d for d in forecast if d.temp_max_C > design_ambient_C + 5]
    if hot_days:
        recs.append(f"{len(hot_days)} day(s) in 7-day forecast exceed "
                    f"design basis by >5°C — plan for reduced capacity.")

    # High humidity warning (relevant for wet-bulb-limited cooling)
    if current.relative_humidity_pct > 85:
        recs.append("High humidity may limit evaporative cooling effectiveness.")

    # Freezing warning
    cold_days = [d for d in forecast if d.temp_min_C < 0]
    if cold_days:
        recs.append(f"{len(cold_days)} day(s) with sub-zero temperatures — "
                    f"check freeze protection for instruments and water systems.")

    return WeatherResult(
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        current=current,
        forecast=forecast,
        cooler_impact=cooler,
        recommendations=recs,
        raw_url=raw_url,
    )


# ---------------------------------------------------------------------------
# Formatter for LLM context
# ---------------------------------------------------------------------------

def format_weather_result(result: WeatherResult) -> str:
    """Format weather analysis for LLM follow-up."""
    lines = [
        f"=== Weather Analysis: {result.location_name} ===",
        f"Coordinates: {result.latitude:.4f}°N, {result.longitude:.4f}°E",
        "",
        "--- Current Conditions ---",
        f"  Temperature:       {result.current.temperature_C:.1f}°C",
        f"  Feels like:        {result.current.apparent_temperature_C:.1f}°C",
        f"  Relative Humidity: {result.current.relative_humidity_pct:.0f}%",
        f"  Wind:              {result.current.wind_speed_m_s:.1f} m/s "
        f"({result.current.wind_direction_deg:.0f}°)",
        f"  Cloud Cover:       {result.current.cloud_cover_pct:.0f}%",
        f"  Precipitation:     {result.current.precipitation_mm:.1f} mm",
    ]

    if result.cooler_impact:
        ci = result.cooler_impact
        lines += [
            "",
            "--- Air-Cooler Impact ---",
            f"  Design Ambient:    {ci.design_ambient_C:.1f}°C",
            f"  Current Ambient:   {ci.current_ambient_C:.1f}°C",
            f"  Delta:             {ci.delta_C:+.1f}°C",
            f"  Capacity Factor:   {ci.capacity_factor:.1%}",
            f"  Status:            {ci.status}",
            f"  {ci.message}",
        ]

    if result.forecast:
        lines += ["", "--- 7-Day Forecast ---"]
        for d in result.forecast:
            lines.append(
                f"  {d.date}: {d.temp_min_C:5.1f} – {d.temp_max_C:5.1f}°C "
                f"(avg {d.temp_avg_C:.1f}°C), "
                f"humidity {d.humidity_avg_pct:.0f}%, "
                f"precip {d.precipitation_sum_mm:.1f}mm"
            )

    if result.recommendations:
        lines += ["", "--- Recommendations ---"]
        for r in result.recommendations:
            lines.append(f"  • {r}")

    return "\n".join(lines)
