import json
import math
import urllib.parse
from hazard_intel import render_hazard_intel
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

WEIGHTS = {
    "Rainfall (24 h)": 30,
    "Rainfall (3 days)": 25,
    "Slope": 25,
    "Soil moisture": 20,
}

NORMALIZATION = {
    "Rainfall (24 h)": 200,
    "Rainfall (3 days)": 400,
    "Slope": 60,
}


def clamp(value, lower=0, upper=1):
    return max(lower, min(value, upper))


def normalize_live_weather_to_inputs(weather):
    if weather is None:
        return {"rain_24h": 0.0, "rain_3d": 0.0, "soil_moisture": 0.0}
    return {
        "rain_24h": float(weather.get("rain_24h", 0.0)),
        "rain_3d": float(weather.get("rain_3d", 0.0)),
        "soil_moisture": float(weather.get("soil_moisture", 0.0)),
    }


def calculate_risk_score(rain_24h, rain_3d, slope, soil_moisture):
    scaled = {
        "Rainfall (24 h)": clamp(rain_24h / NORMALIZATION["Rainfall (24 h)"], 0, 1),
        "Rainfall (3 days)": clamp(rain_3d / NORMALIZATION["Rainfall (3 days)"], 0, 1),
        "Slope": clamp(slope / NORMALIZATION["Slope"], 0, 1),
        "Soil moisture": clamp(soil_moisture / 100, 0, 1),
    }
    contributions = {name: round(scaled[name] * WEIGHTS[name], 1) for name in WEIGHTS}
    score = int(sum(contributions.values()))
    return max(0, min(score, 100))


def classify_risk(score):
    if score < 30:
        return "Low"
    if score < 55:
        return "Moderate"
    if score < 75:
        return "Elevated"
    return "High"


def build_contribution_breakdown(rain_24h, rain_3d, slope, soil_moisture):
    scaled = {
        "Rainfall (24 h)": clamp(rain_24h / NORMALIZATION["Rainfall (24 h)"], 0, 1),
        "Rainfall (3 days)": clamp(rain_3d / NORMALIZATION["Rainfall (3 days)"], 0, 1),
        "Slope": clamp(slope / NORMALIZATION["Slope"], 0, 1),
        "Soil moisture": clamp(soil_moisture / 100, 0, 1),
    }
    return {name: round(scaled[name] * WEIGHTS[name], 1) for name in WEIGHTS}


@st.cache_data(ttl=300)
def fetch_live_weather(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&"
        "hourly=precipitation,soil_moisture_0_1cm&daily=precipitation_sum&past_days=2&forecast_days=1&timezone=auto"
    )
    try:
        request = Request(url, headers={"User-Agent": "MountainWatchAI/1.0"})
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None

    hourly = payload.get("hourly", {})
    daily = payload.get("daily", {})
    hourly_times = hourly.get("time", [])
    hourly_precip = hourly.get("precipitation", [])
    hourly_soil = hourly.get("soil_moisture_0_1cm", [])

    rain_24h = 0.0
    for ts, value in zip(hourly_times, hourly_precip):
        try:
            sample_time = pd.Timestamp(ts)
            if sample_time.tzinfo is None:
                sample_time = sample_time.tz_localize("UTC")
            if sample_time >= pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(hours=24):
                rain_24h += float(value or 0)
        except (TypeError, ValueError):
            continue

    daily_precip = daily.get("precipitation_sum", [])
    rain_3d = 0.0 if not daily_precip else round(sum(float(v or 0) for v in daily_precip[-3:]), 1)

    soil_moisture = 60.0
    if hourly_soil:
        last_soil = hourly_soil[-1]
        if isinstance(last_soil, (int, float)):
            soil_moisture = float(last_soil)

    return {
        "rain_24h": round(rain_24h, 1),
        "rain_3d": round(rain_3d, 1),
        "soil_moisture": min(100.0, max(0.0, soil_moisture)),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


@st.cache_data(ttl=300)
def fetch_recent_earthquakes(lat, lon, radius_km=250):
    url = (
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    )
    try:
        request = Request(url, headers={"User-Agent": "MountainWatchAI/1.0"})
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return pd.DataFrame(columns=["place", "mag", "time", "latitude", "longitude"])

    features = payload.get("features", [])
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None, None])
        if len(coords) < 2:
            continue
        eq_lat, eq_lon = coords[1], coords[0]
        distance = ((eq_lat - lat) ** 2 + (eq_lon - lon) ** 2) ** 0.5 * 111.0
        if distance <= radius_km:
            rows.append({
                "place": props.get("place", "Unknown area"),
                "mag": props.get("mag", 0),
                "time": pd.to_datetime(props.get("time", 0), unit="ms"),
                "latitude": eq_lat,
                "longitude": eq_lon,
            })

    eq_df = pd.DataFrame(rows, columns=["place", "mag", "time", "latitude", "longitude"])
    if eq_df.empty or "mag" not in eq_df.columns:
        return pd.DataFrame(columns=["place", "mag", "time", "latitude", "longitude"])
    return eq_df.sort_values("mag", ascending=False)


def classify_rainfall_signal(peak_hour_mm):
    if peak_hour_mm >= 50:
        return "Extreme hourly rain signal"
    if peak_hour_mm >= 20:
        return "Heavy rain signal"
    return "No heavy rain signal"


@st.cache_data(ttl=600)
def fetch_hazard_outlook(lat, lon, radius_km=250):
    radius = radius_km * 0.75
    latitude_step = radius * 2 / 8 / 111.0
    longitude_step = radius * 2 / 8 / max(111.0 * abs(math.cos(math.radians(lat))), 1.0)
    points = [
        (max(-90.0, min(90.0, lat + row * latitude_step)),
         ((lon + column * longitude_step + 180) % 360) - 180)
        for row in range(-4, 5)
        for column in range(-4, 5)
    ]
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={','.join(f'{point[0]:.4f}' for point in points)}&"
        f"longitude={','.join(f'{point[1]:.4f}' for point in points)}&"
        "hourly=precipitation,wind_gusts_10m,temperature_2m,snowfall,weather_code,cape,cloud_cover&"
        "forecast_days=3&timezone=UTC"
    )
    try:
        request = Request(url, headers={"User-Agent": "MountainWatchAI/1.0"})
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return {"times": [], "points": [], "available": False}

    locations = payload if isinstance(payload, list) else [payload]
    rows = []
    for (point_lat, point_lon), location in zip(points, locations):
        hourly = location.get("hourly", {})
        rows.append({
            "latitude": point_lat,
            "longitude": point_lon,
            "precipitation": hourly.get("precipitation", []),
            "wind": hourly.get("wind_gusts_10m", []),
            "temperature": hourly.get("temperature_2m", []),
            "snowfall": hourly.get("snowfall", []),
            "weather_code": hourly.get("weather_code", []),
            "cape": hourly.get("cape", []),
            "cloud_cover": hourly.get("cloud_cover", []),
        })
    times = locations[0].get("hourly", {}).get("time", []) if locations else []
    available = bool(times and rows and rows[0]["precipitation"] and rows[0]["temperature"])
    return {"times": times, "points": rows, "available": available}


@st.cache_data(ttl=300)
def fetch_radar_frames():
    try:
        request = Request(
            "https://api.rainviewer.com/public/weather-maps.json",
            headers={"User-Agent": "MountainWatchAI/1.0"},
        )
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return []

    frames = payload.get("radar", {}).get("past", []) + payload.get("radar", {}).get("nowcast", [])
    return [{"path": frame.get("path"), "time": frame.get("time")} for frame in frames if frame.get("path")]


def render_hazard_map(lat, lon, radius_km, earthquakes, hazard_outlook, radar_frames):
    earthquake_points = [{
        "latitude": float(event["latitude"]),
        "longitude": float(event["longitude"]),
        "magnitude": float(event["mag"] or 0),
        "place": str(event["place"]),
        "time": str(event["time"]),
    } for event in earthquakes.to_dict("records")]
    event_json = json.dumps(earthquake_points).replace("</", "<\\/")
    forecast_json = json.dumps(hazard_outlook).replace("</", "<\\/")
    radar_json = json.dumps(radar_frames).replace("</", "<\\/")
    map_html = f"""
    <!doctype html>
    <html><head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
        <style>
            html, body, #map {{ height: 100%; width: 100%; margin: 0; font: 13px Arial, sans-serif; }}
            .leaflet-control-layers, .leaflet-bar {{ border: 0 !important; box-shadow: 0 2px 9px #001d2d55 !important; }}
            .hazard-menu {{ background: #075879; padding: 5px; border-radius: 4px; box-shadow: 0 2px 9px #001d2d66; }}
            .hazard-menu button {{ display: block; border: 0; width: 132px; padding: 9px 10px; margin: 2px 0; text-align: left; color: #effaff; background: transparent; cursor: pointer; border-radius: 3px; font-size: 12px; }}
            .hazard-menu button.active, .hazard-menu button:hover {{ background: #f2ab00; color: #162a33; }}
            .time-dock {{ box-sizing: border-box; width: min(510px, calc(100vw - 95px)); background: #075879ee; color: white; padding: 10px 12px; border-radius: 4px; box-shadow: 0 2px 9px #001d2d66; }}
            .time-dock .time-row {{ display: flex; align-items: center; gap: 10px; }}
            .time-dock input {{ flex: 1; min-width: 50px; accent-color: #ffb900; }}
            .time-dock button {{ border: 1px solid #ffffff55; color: white; background: #08769a; border-radius: 3px; padding: 6px 12px; cursor: pointer; }}
            .time-label {{ min-width: 74px; font-variant-numeric: tabular-nums; }}
            .layer-status {{ box-sizing: border-box; min-width: 190px; max-width: min(300px, calc(100vw - 180px)); background: #fff; color: #18313b; padding: 9px 12px; border-radius: 4px; box-shadow: 0 2px 9px #001d2d55; }}
            .layer-status-label {{ display: block; color: #58727b; font-size: 9px; font-weight: 700; margin-bottom: 3px; }}
            .layer-status strong {{ display: block; font-size: 13px; }}
            .layer-status small {{ display: block; margin-top: 3px; color: #58727b; line-height: 1.35; }}
            .hazard-alert {{ animation: alert-pulse 1.4s ease-in-out infinite; }}
            @keyframes alert-pulse {{ 50% {{ stroke-opacity: 0.45; stroke-width: 5; }} }}
            .map-legend {{ background: #fff; padding: 9px 12px; line-height: 1.55; color: #263238; border-radius: 4px; box-shadow: 0 1px 6px #0003; max-width: 200px; }}
            .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
            .cloud-gradient {{ height: 12px; margin: 6px 0 2px; border: 1px solid #b6c2c9; background: linear-gradient(90deg, rgba(239,246,250,0.05), rgba(239,246,250,0.48), rgba(255,255,255,0.94)); }}
            .cloud-range {{ display: flex; justify-content: space-between; font-size: 11px; }}
            .leaflet-popup-content {{ line-height: 1.5; }}
        </style>
    </head><body>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            const map = L.map('map', {{ zoomControl: false }}).setView([{lat}, {lon}], 7);
            const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ maxZoom: 19, attribution: 'Tiles &copy; Esri' }}).addTo(map);
            const street = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }});
            const topo = L.tileLayer('https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 17, attribution: '&copy; OpenTopoMap contributors' }});
            map.createPane('labelsPane');
            map.getPane('labelsPane').style.zIndex = 425;
            map.getPane('labelsPane').style.pointerEvents = 'none';
            const labels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ pane: 'labelsPane', maxZoom: 19, attribution: 'Labels &copy; Esri' }}).addTo(map);
            map.createPane('satelliteCloudPane');
            map.getPane('satelliteCloudPane').style.zIndex = 420;
            map.getPane('satelliteCloudPane').style.pointerEvents = 'none';
            const satelliteCloudLayer = L.tileLayer('https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/Himawari_AHI_Band13_Clean_Infrared/default/GoogleMapsCompatible_Level6/{{z}}/{{y}}/{{x}}.png', {{
                pane: 'satelliteCloudPane', maxNativeZoom: 6, maxZoom: 12, opacity: 0.78,
                attribution: 'Himawari imagery &copy; NASA GIBS'
            }}).addTo(map);
            L.control.zoom({{ position: 'topleft' }}).addTo(map);
            L.control.layers({{ 'Satellite': satellite, 'Street map': street, 'Topographic': topo }}, null, {{ position: 'topright' }}).addTo(map);
            map.createPane('forecastPane');
            map.getPane('forecastPane').style.zIndex = 450;
            map.getPane('forecastPane').style.pointerEvents = 'none';
            const earthquakeData = {event_json};
            const forecast = {forecast_json};
            const radarFrames = {radar_json};
            const safe = value => String(value ?? '').replace(/[&<>"']/g, char => ({{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }}[char]));
            const radarLayer = L.tileLayer('', {{ opacity: 0.78, maxZoom: 12, tileSize: 256, attribution: 'Radar &copy; RainViewer' }});
            const forecastLayer = L.gridLayer({{ tileSize: 256, opacity: 0.93, maxZoom: 12, pane: 'forecastPane' }}).addTo(map);
            const earthquakeLayer = L.layerGroup();
            earthquakeData.forEach(event => {{
                const radius = Math.max(6, Math.min(18, 5 + event.magnitude * 2));
                L.circleMarker([event.latitude, event.longitude], {{ radius, color: '#842029', weight: 2, fillColor: '#dc3545', fillOpacity: 0.8, className: event.magnitude >= 4.5 ? 'hazard-alert' : '' }})
                    .bindPopup(`<strong>M${{event.magnitude.toFixed(1)}} earthquake</strong><br>${{safe(event.place)}}<br>${{safe(event.time)}} UTC`).addTo(earthquakeLayer);
            }});
            L.circleMarker([{lat}, {lon}], {{ radius: 7, color: '#084298', weight: 2, fillColor: '#0d6efd', fillOpacity: 1 }})
                .bindPopup('<strong>Monitoring location</strong>').addTo(map);
            let activeHazard = 'satellite';
            let forecastHour = 0;
            let radarIndex = Math.max(0, radarFrames.length - 1);
            let activeCount = 0;
            let timer = null;
            const hazardNames = {{ satellite: 'Satellite clouds', radar: 'Rain radar', clouds: 'Cloud cover forecast', rain: 'Rain / flood potential', wind: 'Wind gusts', heat: 'Heat stress', cold: 'Cold / snow', storm: 'Thunderstorm risk', quakes: 'Earthquakes (observed)' }};
            const colors = {{ green: '#35a66f', yellow: '#f5ce32', orange: '#f28c28', red: '#d9363e', blue: '#3485d3', pale: '#e5edf4', quiet: '#398f9b' }};
            const forecastTimes = forecast.times || [];
            function reading(values, hour) {{ return Number(values?.[hour] ?? 0); }}
            function forecastValue(point, hour) {{
                const rain = reading(point.precipitation, hour);
                const rain3h = rain + reading(point.precipitation, hour + 1) + reading(point.precipitation, hour + 2);
                const wind = reading(point.wind, hour);
                const temp = reading(point.temperature, hour);
                const snow = reading(point.snowfall, hour);
                const cape = reading(point.cape, hour);
                const code = reading(point.weather_code, hour);
                if (activeHazard === 'clouds') return reading(point.cloud_cover, hour);
                if (activeHazard === 'rain') return rain3h;
                if (activeHazard === 'wind') return wind;
                if (activeHazard === 'heat') return temp;
                if (activeHazard === 'cold') return Math.max(snow, temp <= 0 ? 1 + Math.min(Math.abs(temp), 5) * 0.2 : 0);
                return Math.max(cape / 20, code >= 95 ? 60 + (code - 95) * 20 : 0);
            }}
            function hazardColor(value) {{
                let stops;
                if (activeHazard === 'clouds') return [239, 246, 250, Math.round(Math.max(0, Math.min(0.92, Math.pow(Math.max(0, (value - 4) / 96), 0.58) * 0.92)) * 255)];
                if (activeHazard === 'rain') stops = [[0.2, [68, 164, 219]], [1, [52, 195, 190]], [5, [69, 181, 93]], [20, [246, 219, 63]], [50, [245, 137, 47]], [100, [218, 49, 58]]];
                else if (activeHazard === 'wind') stops = [[0, [48, 151, 211]], [10, [51, 190, 190]], [20, [76, 183, 107]], [35, [241, 211, 53]], [60, [244, 133, 46]], [90, [207, 48, 67]]];
                else if (activeHazard === 'heat') stops = [[-10, [47, 113, 203]], [5, [47, 177, 211]], [15, [56, 185, 147]], [22, [110, 190, 79]], [28, [242, 210, 58]], [35, [241, 130, 42]], [42, [205, 43, 56]]];
                else if (activeHazard === 'cold') stops = [[0, [54, 158, 173]], [0.5, [75, 192, 224]], [1, [59, 132, 218]], [5, [109, 72, 191]]];
                else stops = [[0, [47, 153, 205]], [500, [40, 190, 170]], [1000, [237, 216, 57]], [2000, [242, 135, 45]], [3000, [207, 43, 61]]];
                if (activeHazard === 'rain' && value < 0.2) return [56, 143, 155, 0];
                if (value <= stops[0][0]) return [56, 143, 155, 145];
                for (let index = 1; index < stops.length; index += 1) {{
                    if (value <= stops[index][0]) {{
                        const [lowValue, lowColor] = stops[index - 1];
                        const [highValue, highColor] = stops[index];
                        const ratio = (value - lowValue) / Math.max(highValue - lowValue, 0.001);
                        const rgb = lowColor.map((channel, channelIndex) => Math.round(channel + (highColor[channelIndex] - channel) * ratio));
                        return [...rgb, 205];
                    }}
                }}
                return [...stops[stops.length - 1][1], 230];
            }}
            function isWeatherSignal(value) {{
                if (activeHazard === 'rain') return value >= 1;
                if (activeHazard === 'wind') return value >= 40;
                if (activeHazard === 'heat') return value >= 32;
                if (activeHazard === 'cold') return value >= 0.5;
                if (activeHazard === 'storm') return value >= 50;
                return true;
            }}
            forecastLayer.createTile = function(coords, done) {{
                const tile = document.createElement('canvas');
                tile.width = tile.height = 256;
                const context = tile.getContext('2d');
                if (forecast.available && forecast.points?.length && ['clouds', 'rain', 'wind', 'heat', 'cold', 'storm'].includes(activeHazard)) {{
                    const values = forecast.points.map(point => ({{ lat: point.latitude, lon: point.longitude, value: forecastValue(point, forecastHour) }}));
                    const gridBounds = values.reduce((bounds, sample) => ({{
                        minLat: Math.min(bounds.minLat, sample.lat),
                        maxLat: Math.max(bounds.maxLat, sample.lat),
                        minLon: Math.min(bounds.minLon, sample.lon),
                        maxLon: Math.max(bounds.maxLon, sample.lon)
                    }}), {{ minLat: 90, maxLat: -90, minLon: 180, maxLon: -180 }});
                    const step = 6;
                    for (let y = 0; y < 256; y += step) {{
                        for (let x = 0; x < 256; x += step) {{
                            const location = map.unproject([coords.x * 256 + x + step / 2, coords.y * 256 + y + step / 2], coords.z);
                            if (location.lat < gridBounds.minLat || location.lat > gridBounds.maxLat || location.lng < gridBounds.minLon || location.lng > gridBounds.maxLon) continue;
                            let weightedValue = 0;
                            let totalWeight = 0;
                            let nearestDistance = Infinity;
                            values.forEach(sample => {{
                                const northKm = (location.lat - sample.lat) * 111;
                                const eastKm = (location.lng - sample.lon) * 111 * Math.cos(location.lat * Math.PI / 180);
                                const distanceSquared = northKm * northKm + eastKm * eastKm;
                                nearestDistance = Math.min(nearestDistance, Math.sqrt(distanceSquared));
                                const weight = 1 / (distanceSquared + 2500);
                                weightedValue += sample.value * weight;
                                totalWeight += weight;
                            }});
                            const interpolated = weightedValue / totalWeight;
                            if (nearestDistance > {radius_km} * 0.45 || !isWeatherSignal(interpolated)) continue;
                            const color = hazardColor(interpolated);
                            const edgeFade = Math.max(0, Math.min(1, (1 - nearestDistance / ({radius_km} * 0.45)) * 1.8));
                            context.fillStyle = `rgba(${{color[0]}},${{color[1]}},${{color[2]}},${{color[3] / 255}})`;
                            context.globalAlpha = edgeFade;
                            context.fillRect(x, y, step, step);
                            context.globalAlpha = 1;
                        }}
                    }}
                }}
                tile.classList.add('leaflet-tile-loaded');
                done(null, tile);
                return tile;
            }};
            function assess(point, hour) {{
                const rain = reading(point.precipitation, hour);
                const rain3h = rain + reading(point.precipitation, hour + 1) + reading(point.precipitation, hour + 2);
                const wind = reading(point.wind, hour);
                const temp = reading(point.temperature, hour);
                const snow = reading(point.snowfall, hour);
                const cape = reading(point.cape, hour);
                const code = reading(point.weather_code, hour);
                if (activeHazard === 'clouds') {{
                    const cloudCover = reading(point.cloud_cover, hour);
                    return {{ active: false, value: `${{cloudCover.toFixed(0)}}% cloud cover`, detail: 'Open-Meteo model outlook', color: '#dfe9ef', level: 'Forecast cloud cover' }};
                }}
                if (activeHazard === 'rain') return {{ active: rain >= 5 || rain3h >= 20, value: `${{rain.toFixed(1)}} mm/h`, detail: `${{rain3h.toFixed(1)}} mm / 3h`, color: rain >= 50 || rain3h >= 100 ? colors.red : rain >= 20 || rain3h >= 50 ? colors.orange : rain >= 5 || rain3h >= 20 ? colors.yellow : colors.quiet, level: rain >= 50 || rain3h >= 100 ? 'Extreme rain / flood potential proxy' : rain >= 20 || rain3h >= 50 ? 'Heavy rain signal' : rain >= 5 || rain3h >= 20 ? 'Rainfall signal' : 'No heavy-rain threshold crossed' }};
                if (activeHazard === 'wind') return {{ active: wind >= 40, value: `${{wind.toFixed(0)}} km/h gust`, detail: 'Forecast peak gust', color: wind >= 90 ? colors.red : wind >= 60 ? colors.orange : wind >= 40 ? colors.yellow : colors.quiet, level: wind >= 90 ? 'Severe gust signal' : wind >= 60 ? 'Strong gust signal' : wind >= 40 ? 'Elevated gust signal' : 'No strong-gust threshold crossed' }};
                if (activeHazard === 'heat') return {{ active: temp >= 32, value: `${{temp.toFixed(1)}} °C`, detail: 'Forecast air temperature', color: temp >= 40 ? colors.red : temp >= 35 ? colors.orange : temp >= 32 ? colors.yellow : colors.quiet, level: temp >= 40 ? 'Extreme heat signal' : temp >= 35 ? 'High heat signal' : temp >= 32 ? 'Elevated heat signal' : 'No heat threshold crossed' }};
                if (activeHazard === 'cold') return {{ active: snow >= 1 || temp <= 0, value: `${{temp.toFixed(1)}} °C`, detail: `${{snow.toFixed(1)}} mm/h snowfall`, color: snow >= 5 ? colors.red : snow >= 1 || temp <= 0 ? colors.blue : colors.quiet, level: snow >= 5 ? 'Heavy snow signal' : snow >= 1 || temp <= 0 ? 'Freezing / snow signal' : 'No freezing or snow threshold crossed' }};
                const thunder = code >= 95 || (cape >= 1000 && rain >= 1);
                return {{ active: thunder, value: `${{cape.toFixed(0)}} J/kg CAPE`, detail: `Rain ${{rain.toFixed(1)}} mm/h; WMO code ${{code}}`, color: code >= 96 || (cape >= 2000 && rain >= 10) ? colors.red : thunder ? colors.orange : colors.quiet, level: thunder ? 'Thunderstorm conditions indicated' : 'No thunderstorm threshold crossed' }};
            }}
            function drawForecast() {{
                activeCount = 0;
                if (!forecast.available || !forecast.points?.length || ['satellite', 'radar', 'quakes'].includes(activeHazard)) return;
                forecast.points.forEach(point => {{ if (assess(point, forecastHour).active) activeCount += 1; }});
                forecastLayer.redraw();
            }}
            function showRadar() {{
                if (!radarFrames.length) return;
                const frame = radarFrames[radarIndex];
                radarLayer.setUrl(`https://tilecache.rainviewer.com${{frame.path}}/256/{{z}}/{{x}}/{{y}}/2/1_1.png`);
                if (!map.hasLayer(radarLayer)) radarLayer.addTo(map);
            }}
            function updateTime() {{
                const slider = document.getElementById('time-slider');
                const label = document.getElementById('time-label');
                const radarMode = activeHazard === 'radar';
                const satelliteMode = activeHazard === 'satellite';
                const max = radarMode ? Math.max(0, radarFrames.length - 1) : Math.max(0, forecastTimes.length - 1);
                slider.max = max;
                slider.disabled = satelliteMode;
                slider.value = radarMode ? radarIndex : forecastHour;
                if (radarMode) {{
                    const timestamp = radarFrames[radarIndex]?.time;
                    label.textContent = timestamp ? new Date(timestamp * 1000).toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit' }}) : 'Radar unavailable';
                    showRadar();
                }} else if (satelliteMode) {{
                    label.textContent = 'Latest satellite image';
                }} else {{
                    const timestamp = forecastTimes[forecastHour];
                    label.textContent = timestamp ? new Date(timestamp + 'Z').toLocaleString([], {{ weekday: 'short', hour: '2-digit', minute: '2-digit' }}) : 'Forecast unavailable';
                    drawForecast();
                }}
                const signalCount = activeHazard === 'quakes' ? earthquakeData.length : activeCount;
                document.getElementById('map-title').textContent = ['radar', 'clouds', 'satellite'].includes(activeHazard) ? hazardNames[activeHazard] : `${{hazardNames[activeHazard]}} · ${{signalCount}} signals`;
                document.getElementById('time-range').textContent = radarMode ? 'Radar observations / nowcast' : satelliteMode ? 'Himawari infrared · NASA GIBS' : 'Forecast hour';
                document.getElementById('layer-status-name').textContent = hazardNames[activeHazard];
                let layerDetail;
                if (satelliteMode) layerDetail = 'Observed Himawari infrared image - latest available';
                else if (radarMode) layerDetail = radarFrames.length ? `RainViewer observed radar - ${{label.textContent}}` : 'Radar frames unavailable';
                else if (activeHazard === 'quakes') layerDetail = earthquakeData.length ? `${{earthquakeData.length}} USGS events in the last 24 hours` : 'No USGS events in the selected radius in the last 24 hours';
                else if (activeHazard === 'clouds') layerDetail = `Open-Meteo cloud forecast - ${{label.textContent}} UTC`;
                else layerDetail = activeCount ? `${{activeCount}} grid points cross the threshold - ${{label.textContent}} UTC` : `No threshold signals at ${{label.textContent}} UTC`;
                document.getElementById('layer-detail').textContent = layerDetail;
                const legend = document.querySelector('.map-legend');
                legend.style.display = ['clouds', 'satellite'].includes(activeHazard) ? 'block' : 'none';
                legend.innerHTML = satelliteMode
                    ? '<strong>Satellite infrared</strong><br><small>Himawari · latest available</small>'
                    : '<strong>Cloud cover forecast</strong><div class="cloud-gradient"></div><div class="cloud-range"><span>Clear</span><span>Overcast</span></div><small>Open-Meteo model outlook</small>';
            }}
            map.on('click', event => {{
                if (!['clouds', 'rain', 'wind', 'heat', 'cold', 'storm'].includes(activeHazard) || !forecast.points?.length) return;
                const nearest = forecast.points.reduce((best, point) => {{
                    const distance = Math.hypot(event.latlng.lat - point.latitude, (event.latlng.lng - point.longitude) * Math.cos(event.latlng.lat * Math.PI / 180));
                    return !best || distance < best.distance ? {{ point, distance }} : best;
                }}, null);
                if (!nearest) return;
                const point = nearest.point;
                const result = assess(point, forecastHour);
                L.popup().setLatLng(event.latlng).setContent(`<strong>${{safe(result.level)}}</strong><br>${{safe(result.value)}}<br>${{safe(result.detail)}}<br>${{safe(forecastTimes[forecastHour] || '')}} UTC<br>Model grid point: ${{point.latitude.toFixed(2)}}, ${{point.longitude.toFixed(2)}}`).openOn(map);
            }});
            function selectHazard(name) {{
                activeHazard = name;
                document.querySelectorAll('.hazard-menu button').forEach(button => button.classList.toggle('active', button.dataset.mode === name));
                if (name === 'quakes') {{
                    if (map.hasLayer(radarLayer)) map.removeLayer(radarLayer);
                    if (map.hasLayer(satelliteCloudLayer)) map.removeLayer(satelliteCloudLayer);
                    if (map.hasLayer(forecastLayer)) map.removeLayer(forecastLayer);
                    if (!map.hasLayer(earthquakeLayer)) earthquakeLayer.addTo(map);
                }} else {{
                    if (map.hasLayer(earthquakeLayer)) map.removeLayer(earthquakeLayer);
                    if (name === 'radar') {{
                        if (map.hasLayer(satelliteCloudLayer)) map.removeLayer(satelliteCloudLayer);
                        if (map.hasLayer(forecastLayer)) map.removeLayer(forecastLayer);
                        if (!map.hasLayer(radarLayer)) radarLayer.addTo(map);
                    }} else if (name === 'satellite') {{
                        if (map.hasLayer(radarLayer)) map.removeLayer(radarLayer);
                        if (map.hasLayer(forecastLayer)) map.removeLayer(forecastLayer);
                        if (!map.hasLayer(satelliteCloudLayer)) satelliteCloudLayer.addTo(map);
                    }} else {{
                        if (map.hasLayer(satelliteCloudLayer)) map.removeLayer(satelliteCloudLayer);
                        if (map.hasLayer(radarLayer)) map.removeLayer(radarLayer);
                        if (!map.hasLayer(forecastLayer)) forecastLayer.addTo(map);
                    }}
                }}
                updateTime();
            }}
            const menu = L.control({{ position: 'topleft' }});
            menu.onAdd = () => {{
                const div = L.DomUtil.create('div', 'hazard-menu');
                [['satellite','Satellite clouds'],['radar','Rain radar'],['clouds','Cloud forecast'],['rain','Rain / flood'],['wind','Wind gusts'],['heat','Heat stress'],['cold','Cold / snow'],['storm','Thunderstorm'],['quakes','Earthquakes']].forEach(([mode,label]) => {{
                    const button = L.DomUtil.create('button', '', div);
                    button.type = 'button'; button.dataset.mode = mode; button.textContent = label;
                    button.classList.toggle('active', mode === activeHazard);
                    button.onclick = () => selectHazard(mode);
                }});
                L.DomEvent.disableClickPropagation(div);
                return div;
            }};
            menu.addTo(map);
            const dock = L.control({{ position: 'bottomleft' }});
            dock.onAdd = () => {{
                const div = L.DomUtil.create('div', 'time-dock');
                div.innerHTML = '<div class="time-row"><strong id="map-title">Rain radar</strong><span style="flex:1"></span><span id="time-range">Past radar / nowcast</span></div><div class="time-row"><span id="time-label" class="time-label">--:--</span><input id="time-slider" type="range" min="0" max="0" value="0"><button id="play-button" type="button" title="Play timeline">Play</button></div>';
                L.DomEvent.disableClickPropagation(div);
                return div;
            }};
            dock.addTo(map);
            const statusControl = L.control({{ position: 'topright' }});
            statusControl.onAdd = () => {{
                const div = L.DomUtil.create('div', 'layer-status');
                div.innerHTML = '<span class="layer-status-label">ACTIVE MAP LAYER</span><strong id="layer-status-name">Satellite clouds</strong><small id="layer-detail">Observed Himawari infrared image - latest available</small>';
                L.DomEvent.disableClickPropagation(div);
                return div;
            }};
            statusControl.addTo(map);
            setTimeout(() => {{
                const slider = document.getElementById('time-slider');
                slider.oninput = () => {{ if (activeHazard === 'radar') radarIndex = Number(slider.value); else forecastHour = Number(slider.value); updateTime(); }};
                document.getElementById('play-button').onclick = event => {{
                    if (timer) {{ clearInterval(timer); timer = null; event.target.textContent = 'Play'; return; }}
                    event.target.textContent = 'Pause';
                    timer = setInterval(() => {{
                        const max = activeHazard === 'radar' ? radarFrames.length - 1 : forecastTimes.length - 1;
                        if (max < 1) {{ clearInterval(timer); timer = null; event.target.textContent = 'Play'; return; }}
                        if (activeHazard === 'radar') radarIndex = (radarIndex + 1) % (max + 1); else forecastHour = (forecastHour + 1) % (max + 1);
                        updateTime();
                    }}, activeHazard === 'radar' ? 650 : 900);
                }};
                updateTime();
            }}, 0);
            L.control.scale({{ imperial: false }}).addTo(map);
            const cloudLegend = L.control({{ position: 'bottomright' }});
            cloudLegend.onAdd = () => {{
                const div = L.DomUtil.create('div', 'map-legend');
                div.innerHTML = '<strong>Cloud cover forecast</strong><div class="cloud-gradient"></div><div class="cloud-range"><span>Clear</span><span>Overcast</span></div><small>Model outlook</small>';
                return div;
            }};
            cloudLegend.addTo(map);
            document.querySelector('.map-legend').style.display = 'block';
            document.querySelector('.map-legend').innerHTML = '<strong>Satellite infrared</strong><br><small>Himawari · latest available</small>';
        </script>
    </body></html>
    """
    st.components.v1.html(map_html, height=650, scrolling=False)


@st.cache_data(ttl=300)
def fetch_disaster_feed(limit=8):
    url = f"https://eonet.gsfc.nasa.gov/api/v3/events?status=open&limit={limit}"
    try:
        request = Request(url, headers={"User-Agent": "MountainWatchAI/1.0"})
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return []
    return payload.get("events", [])


def determine_alert_level(score, severe_quakes, rainfall_24h):
    if score >= 80 or severe_quakes >= 1 or rainfall_24h >= 120:
        return "ALERT"
    if score >= 60 or severe_quakes >= 1 or rainfall_24h >= 80:
        return "WATCH"
    if score >= 40 or rainfall_24h >= 50:
        return "MONITOR"
    return "NORMAL"


def render_alert_badge(alert_level):
    colors = {
        "ALERT": "#d32f2f",
        "WATCH": "#f57c00",
        "MONITOR": "#1976d2",
        "NORMAL": "#2e7d32",
    }
    labels = {
        "ALERT": "🚨 ALERT",
        "WATCH": "⚠️ WATCH",
        "MONITOR": "📡 MONITOR",
        "NORMAL": "✅ NORMAL",
    }
    color = colors.get(alert_level, "#2e7d32")
    label = labels.get(alert_level, "✅ NORMAL")
    st.markdown(
        f"""
        <div style="
            background: {color};
            color: white;
            display: inline-block;
            padding: 10px 18px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 14px;
            letter-spacing: 0.5px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            margin-bottom: 16px;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_recommendations(alert_level):
    if alert_level == "ALERT":
        return [
            "Move to higher ground if rivers are rising.",
            "Avoid slopes, culverts, and road cuts after heavy rain.",
            "Keep emergency contacts and essentials ready.",
        ]
    if alert_level == "WATCH":
        return [
            "Stay alert and monitor official advisories.",
            "Prepare an evacuation route and emergency kit.",
            "Avoid travel through unstable terrain.",
        ]
    if alert_level == "MONITOR":
        return [
            "Monitor conditions closely.",
            "Keep mobile signal and power backup ready.",
            "Check weather updates before travel.",
        ]
    return [
        "Current conditions are relatively stable.",
        "Use caution near slopes and drainage lines.",
        "Continue monitoring official advisories.",
    ]


def sanitize_whatsapp_number(number):
    digits = "".join(ch for ch in str(number) if ch.isdigit())
    return digits


def build_whatsapp_link(phone_number, message):
    cleaned = sanitize_whatsapp_number(phone_number)
    if not cleaned:
        return ""
    encoded = urllib.parse.quote(message)
    return f"https://wa.me/{cleaned}?text={encoded}"


def load_historical_context(csv_path=None):
    if csv_path is None:
        csv_path = Path(__file__).with_name("landslides.csv")

    if not Path(csv_path).exists():
        return {
            "top_trigger": "Not available",
            "trigger_counts": pd.DataFrame({"Trigger": ["Not available"], "Events": [0]}),
        }

    df = pd.read_csv(csv_path)
    trigger_series = df["landslide_trigger"].fillna("unknown").astype(str)
    trigger_counts = trigger_series.str.replace("_", " ").str.title().value_counts()
    return {
        "top_trigger": trigger_counts.index[0] if not trigger_counts.empty else "Unknown",
        "trigger_counts": trigger_counts.rename_axis("Trigger").reset_index(name="Events").head(5),
    }


def main():
    st.set_page_config(page_title="MountainWatch AI", page_icon="⛰️", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] { padding-top: 0.7rem; }
        [data-testid="stElementContainer"]:has(.mw-map-header) { margin-bottom: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Monitoring settings")
    use_live_data = st.sidebar.toggle("Use live weather data", value=True)
    lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=30.3165, step=0.0001)
    lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.0322, step=0.0001)
    radius_km = st.sidebar.slider("Earthquake radius search (km)", 50, 500, 250)

    st.sidebar.subheader("WhatsApp alerts")
    enable_whatsapp = st.sidebar.toggle("Enable WhatsApp alerting", value=False)
    whatsapp_number = st.sidebar.text_input("WhatsApp number", value="", placeholder="919876543210")
    whatsapp_template = st.sidebar.text_area(
        "Alert message",
        value="🚨 MountainWatch AI alert: High-risk conditions detected near your location. Please check official advisories and prepare for safety.",
        height=120,
    )

    if use_live_data:
        live_weather = fetch_live_weather(lat, lon)
        if live_weather:
            st.sidebar.success(f"Live weather refreshed at {live_weather['timestamp']}")
            rain_24h = live_weather["rain_24h"]
            rain_3d = live_weather["rain_3d"]
            soil = live_weather["soil_moisture"]
            data_source = "Open-Meteo live weather"
        else:
            st.sidebar.warning("Live weather unavailable. Using manual fallback inputs.")
            rain_24h = st.sidebar.slider("Rainfall, last 24 h (mm)", 0, 300, 70)
            rain_3d = st.sidebar.slider("Rainfall, last 3 days (mm)", 0, 600, 150)
            soil = st.sidebar.slider("Soil moisture (%)", 0, 100, 60)
            data_source = "Manual fallback"
    else:
        rain_24h = st.sidebar.slider("Rainfall, last 24 h (mm)", 0, 300, 70)
        rain_3d = st.sidebar.slider("Rainfall, last 3 days (mm)", 0, 600, 150)
        soil = st.sidebar.slider("Soil moisture (%)", 0, 100, 60)
        data_source = "Manual input"

    slope = st.sidebar.slider("Slope angle (degrees)", 0, 70, 35)
    refresh_map = st.sidebar.button("Refresh map data", icon="🔄")

    contrib = build_contribution_breakdown(rain_24h, rain_3d, slope, soil)
    score = calculate_risk_score(rain_24h, rain_3d, slope, soil)
    level = classify_risk(score)
    eq_df = fetch_recent_earthquakes(lat, lon, radius_km)
    severe_quakes = int((eq_df["mag"] >= 4.5).sum()) if not eq_df.empty else 0
    alert_level = determine_alert_level(score, severe_quakes, rain_24h)
    recommendations = get_recommendations(alert_level)

    st.markdown(
        '<div class="mw-map-header" style="display:flex;align-items:baseline;gap:12px;min-height:38px;">'
        '<strong style="font-size:24px;color:#075879;">MountainWatch</strong>'
        '<span style="font-size:11px;font-weight:700;color:#52717b;letter-spacing:1px;">LIVE HAZARD MAP</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if refresh_map:
        fetch_recent_earthquakes.clear()
        fetch_hazard_outlook.clear()
        fetch_radar_frames.clear()
        st.rerun()

    hazard_outlook = fetch_hazard_outlook(lat, lon, radius_km)
    radar_frames = fetch_radar_frames()
    if not hazard_outlook["available"]:
        st.warning("Regional weather forecast is unavailable. Radar and earthquake layers may still be available.")
    if not radar_frames:
        st.info("Live radar frames are currently unavailable from the radar provider; forecast overlays remain available.")
    render_hazard_map(lat, lon, radius_km, eq_df, hazard_outlook, radar_frames)
    st.caption(
        "Satellite clouds are Himawari infrared observations from NASA GIBS; Rain radar is RainViewer radar/nowcast. "
        "Other colored hazard layers are localized Open-Meteo forecasts, not observations. They are not official warnings "
        "and can miss local mountain conditions."
    )

    if enable_whatsapp and whatsapp_number:
        whatsapp_link = build_whatsapp_link(whatsapp_number, whatsapp_template)
        st.sidebar.markdown(
            f"[Send alert on WhatsApp]( {whatsapp_link} )",
            unsafe_allow_html=True,
        )

    render_alert_badge(alert_level)

    if enable_whatsapp and whatsapp_number and alert_level in {"ALERT", "WATCH", "MONITOR"}:
        st.markdown(
            f"<a href=\"{build_whatsapp_link(whatsapp_number, whatsapp_template)}\" target=\"_blank\"><button style='background:#25D366;color:white;padding:10px 18px;border:none;border-radius:12px;font-weight:700;cursor:pointer;'>Send WhatsApp alert</button></a>",
            unsafe_allow_html=True,
        )

    if "last_alert_level" not in st.session_state:
        st.session_state.last_alert_level = None

    if alert_level in {"ALERT", "WATCH"} and st.session_state.last_alert_level != alert_level:
        if alert_level == "ALERT":
            st.toast("🚨 ALERT: evacuation preparation may be required. Monitor official advisories.")
        else:
            st.toast("⚠️ WATCH: local conditions are worsening. Stay alert.")
        st.session_state.last_alert_level = alert_level
    elif alert_level == "NORMAL" and st.session_state.last_alert_level not in {None, "NORMAL"}:
        st.toast("✅ Conditions have returned to a more stable state.")
        st.session_state.last_alert_level = "NORMAL"

    if alert_level == "ALERT":
        st.warning("🚨 ALERT: high danger conditions detected. People in the area should prepare to evacuate if local authorities advise it.")
    elif alert_level == "WATCH":
        st.warning("⚠️ WATCH: elevated risk. Stay alert and monitor local updates.")
    elif alert_level == "MONITOR":
        st.info("📡 MONITOR: conditions are being watched. Keep updates ready.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk score", f"{score}/100")
    col2.metric("Risk level", level)
    col3.metric("Rain 24h", f"{rain_24h} mm")
    col4.metric("Earthquakes", severe_quakes)

    st.subheader("Live situational awareness")
    a, b, c = st.columns(3)
    a.metric("Rain 3d", f"{rain_3d} mm")
    b.metric("Soil moisture", f"{soil:.1f}%")
    c.metric("Data source", data_source)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live hazards", "Risk breakdown", "Safety actions"])

    with tab1:
        st.write("This dashboard combines live weather, recent earthquake activity, and historical landslide patterns to support situational awareness.")
        if not eq_df.empty:
            st.subheader("Recent earthquake activity")
            st.dataframe(eq_df[["place", "mag", "time"]].head(10), use_container_width=True)
        else:
            st.caption("No significant earthquake activity was found in the selected radius in the past day.")

        st.subheader("Global disaster feed")
        events = fetch_disaster_feed(8)
        if events:
            for event in events:
                title = event.get("title", "Disaster alert")
                categories = ", ".join(item.get("title", "") for item in event.get("categories", []))
                st.markdown(f"- **{title}** — {categories}")
        else:
            st.caption("No live disaster feed data available right now.")

    with tab2:
       render_hazard_intel(lat, lon)

    with tab3:
        factor_df = pd.DataFrame({"Factor": list(contrib), "Points": list(contrib.values())}).set_index("Factor")
        st.bar_chart(factor_df)
        top = max(contrib, key=contrib.get)
        st.write(f"Biggest contributor: **{top}** ({contrib[top]} points).")

    with tab4:
        st.subheader("Recommended action plan")
        for item in recommendations:
            st.write(f"- {item}")

        st.info(
            "Safety: always check official advisories from IMD, NDMA, local authorities, and emergency services. "
            "This prototype is for awareness and not an official emergency warning system."
        )

    st.divider()
    st.subheader("Historical landslide context")
    context = load_historical_context()
    st.write(f"The most common trigger in the historical landslide dataset is **{context['top_trigger']}**.")
    st.bar_chart(context["trigger_counts"].set_index("Trigger"))


if __name__ == "__main__":
    main()