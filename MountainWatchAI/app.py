import json
import urllib.parse
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
    st.set_page_config(page_title="MountainWatch AI", page_icon="⛰️", layout="wide")
    st.title("⛰️ MountainWatch AI")
    st.caption("Live disaster awareness for landslides, earthquakes, and extreme rainfall.")

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

    contrib = build_contribution_breakdown(rain_24h, rain_3d, slope, soil)
    score = calculate_risk_score(rain_24h, rain_3d, slope, soil)
    level = classify_risk(score)
    eq_df = fetch_recent_earthquakes(lat, lon, radius_km)
    severe_quakes = int((eq_df["mag"] >= 4.5).sum()) if not eq_df.empty else 0
    alert_level = determine_alert_level(score, severe_quakes, rain_24h)
    recommendations = get_recommendations(alert_level)

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
        st.write("Natural hazards tracked in this prototype: landslides, earthquakes, heavy rainfall, and flood risk indicators.")
        st.map(eq_df[["latitude", "longitude"]] if not eq_df.empty else pd.DataFrame(columns=["latitude", "longitude"]))

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