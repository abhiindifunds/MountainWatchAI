"""hazard_intel.py - alert zones, past/now/future timeline, landslide log, stay finder."""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
ZONE_KM = 15
HOTSPOTS = [
    ("Chamoli", 30.4081, 79.3194), ("Mussoorie", 30.4598, 78.0644),
    ("Nainital", 29.3919, 79.4542), ("Rishikesh", 30.0869, 78.2676),
    ("Rudraprayag", 30.2844, 78.9811), ("Uttarkashi", 30.7268, 78.4354),
    ("Pithoragarh", 29.5829, 80.2182), ("Dehradun", 30.3165, 78.0322),
    ("Nahan (HP)", 30.5598, 77.2954), ("Shimla (HP)", 31.1048, 77.1734),
]


def _get_json(url, data=None, timeout=25):
    try:
        req = Request(url, data=data, headers={
            "User-Agent": "MountainWatchAI/1.0",
            "Accept": "application/json",
        })
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _km(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(x))


def level_for(rain_6h, peak_hour, cape):
    """0 Low, 1 Moderate, 2 High, 3 Severe. Thresholds are assumptions: tune them."""
    cloudburst = peak_hour >= 50 or (peak_hour >= 30 and cape >= 800)
    lvl = 3 if cloudburst or rain_6h >= 60 else 2 if rain_6h >= 30 else 1 if rain_6h >= 15 else 0
    return lvl, cloudburst


def rain_window_stats(rain, cape):
    if not rain:
        return 0.0, 0.0, 0.0
    peak_index = max(range(len(rain)), key=rain.__getitem__)
    peak_hour = float(rain[peak_index] or 0)
    peak_cape = float(cape[peak_index] or 0) if peak_index < len(cape) else 0.0
    return sum(float(value or 0) for value in rain), peak_hour, peak_cape


@st.cache_data(ttl=600)
def fetch_forecast():
    url = (
        "https://api.open-meteo.com/v1/forecast?latitude=" + ",".join(str(h[1]) for h in HOTSPOTS)
        + "&longitude=" + ",".join(str(h[2]) for h in HOTSPOTS)
        + "&hourly=precipitation,cape&past_days=1&forecast_days=2&timezone=UTC"
    )
    data = _get_json(url)
    if data is None:
        return None
    return data if isinstance(data, list) else [data]


def build_zones(raw, demo=False):
    now = datetime.now(UTC).replace(tzinfo=None)
    zones = []
    for (name, lat, lon), d in zip(HOTSPOTS, raw or []):
        hr = d.get("hourly", {})
        times = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in hr.get("time", [])]
        p = [float(v or 0) for v in hr.get("precipitation", [])]
        c = [float(v or 0) for v in hr.get("cape", [])]
        past = [i for i, t in enumerate(times) if t <= now]
        if not past or len(p) != len(times) or len(c) != len(times):
            continue
        i0 = past[-1]
        if demo and name.startswith("Nahan"):  # simulated cloudburst signal
            for k, v in ((1, 65.0), (2, 40.0)):
                if i0 + k < len(p):
                    p[i0 + k], c[i0 + k] = v, 1800.0

        def win(a, b):
            a, b = max(0, a), min(len(p), b)
            return (p[a:b] or [0]), (c[a:b] or [0])
        pn, cn = win(i0 - 5, i0 + 1)
        pf, cf = win(i0 + 1, i0 + 7)
        l_now, cb_now = level_for(*rain_window_stats(pn, cn))
        l_next, cb_next = level_for(*rain_window_stats(pf, cf))
        lo, hi = max(0, i0 - 24), min(len(p), i0 + 37)
        zones.append(dict(
            name=name, lat=lat, lon=lon, offs=list(range(lo - i0, hi - i0)),
            p=[round(v, 1) for v in p[lo:hi]], c=[round(v) for v in c[lo:hi]],
            rain_24h=round(sum(p[max(0, i0 - 23):i0 + 1]), 1), next6=round(sum(pf), 1),
            level=max(l_now, l_next), cb=cb_now or cb_next,
        ))
    return zones


def load_incidents(demo=False):
    """Landslide log: incidents.csv + entries added in the app (+ clearly labelled demo events)."""
    rows = list(st.session_state.get("mw_incidents", []))
    f = Path(__file__).with_name("incidents.csv")  # columns: time_ist,place,lat,lon,source
    if f.exists():
        try:
            df = pd.read_csv(f)
            for r in df.itertuples():
                t = pd.to_datetime(r.time_ist, errors="coerce")
                if pd.notna(t):
                    rows.append(dict(place=str(r.place), lat=float(r.lat), lon=float(r.lon),
                                     t=t.to_pydatetime().replace(tzinfo=IST), source=str(r.source)))
        except Exception:
            st.warning("incidents.csv could not be read. Check its columns: time_ist,place,lat,lon,source")
    if demo:
        now = datetime.now(IST)
        for mins, place, lat, lon in ((50, "Chamoli highway", 30.42, 79.33),
                                      (35, "Mussoorie highway", 30.46, 78.08),
                                      (15, "Nainital highway", 29.40, 79.47)):
            rows.append(dict(place=place + " (DEMO)", lat=lat, lon=lon,
                             t=now - timedelta(minutes=mins), source="Simulated demo event"))
    now = datetime.now(IST)
    return [r for r in rows if timedelta(0) <= now - r["t"] <= timedelta(hours=48)]


def briefing(zones, incidents):
    now = datetime.now(IST)
    recent = sorted((i for i in incidents if now - i["t"] <= timedelta(hours=1)), key=lambda i: i["t"])
    lines = []
    if recent:
        lines.append(f"**{len(recent)} landslide(s) logged in the last hour:** "
                     + "; ".join(f"{i['place']} at {i['t']:%I:%M %p}" for i in recent))
    else:
        lines.append("No landslides logged in the last hour (only logged reports appear here).")
    wet = [z["name"] for z in zones if z["next6"] >= 30]
    if wet:
        lines.append("Heavy rain expected in the next 6 h near **" + ", ".join(wet)
                     + "**, so the chance of further landslides is higher.")
    return lines


def police_banner(places, demo):
    st.markdown(
        "<style>@keyframes mwp{0%,49%{background:#d32f2f}50%,100%{background:#1565c0}}"
        ".mwp{animation:mwp .8s infinite;color:#fff;padding:12px 16px;border-radius:8px;font-weight:700}</style>"
        f"<div class='mwp'>🚨 CLOUDBURST SIGNAL{' (SIMULATED DEMO)' if demo else ''}: "
        f"{', '.join(places)}. Model-based signal, not an official warning. Follow IMD / SDMA advisories.</div>",
        unsafe_allow_html=True)


MAP_HTML = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body,#map{height:100%;margin:0;font:13px Arial,sans-serif}
.panel{position:absolute;z-index:1000;left:50px;bottom:18px;right:12px;max-width:560px;background:#075879ee;color:#fff;padding:10px 12px;border-radius:6px}
.panel input{width:100%;accent-color:#ffb900}
.legend{position:absolute;z-index:1000;right:12px;top:12px;background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 6px #0004;line-height:1.6}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
#cb{display:none;position:absolute;z-index:1000;left:60px;top:10px;max-width:55%;color:#fff;font-weight:700;padding:8px 14px;border-radius:6px;animation:pl .8s infinite}
.police{width:22px;height:22px;border-radius:50%;animation:pl .8s infinite;box-shadow:0 0 14px 6px currentColor}
@keyframes pl{0%,49%{background:#d32f2f;color:#d32f2f}50%,100%{background:#1565c0;color:#1565c0}}
.pulse{animation:pu 1s infinite}@keyframes pu{50%{stroke-opacity:.2;stroke-width:6}}
</style></head><body><div id="map"></div><div id="cb"></div>
<div class="legend"><b>Alert zones</b><br><span class="dot" style="background:#2e9e5b"></span>Low<br><span class="dot" style="background:#f2c230"></span>Moderate<br><span class="dot" style="background:#f28c28"></span>High<br><span class="dot" style="background:#d9363e"></span>Severe<br><b>Landslides</b><br><span class="dot" style="background:#d32f2f"></span>&lt; 1 h ago<br><span class="dot" style="background:#f28c28"></span>1-6 h<br><span class="dot" style="background:#888"></span>older</div>
<div class="panel"><b id="lab"></b> <span id="sub"></span><input id="s" type="range" min="-24" max="36" value="0" step="1"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const D=__DATA__;
const map=L.map('map').setView([30.45,78.5],8);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'&copy; OpenStreetMap'}).addTo(map);
if(D.zones.length)map.fitBounds(D.zones.map(z=>[z.lat,z.lon]),{padding:[40,40]});
const C=['#2e9e5b','#f2c230','#f28c28','#d9363e'],N=['Low','Moderate','High','Severe'];
const zl=L.layerGroup().addTo(map),il=L.layerGroup().addTo(map);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function lvl(s,pk,cp){const cb=pk>=50||(pk>=30&&cp>=800);return[cb||s>=60?3:s>=30?2:s>=15?1:0,cb]}
function clock(t){return new Date(D.now+t*36e5).toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit'})}
function lab(t){return(t==0?'Now':(t>0?'+':'')+t+' h')+' \\u00b7 '+clock(t)+' IST'}
function draw(t){
  zl.clearLayers();il.clearLayers();const cbs=[];let hi=0,shown=0;
  D.zones.forEach(z=>{
    let s=0,pk=0,cp=0;
    z.offs.forEach((o,i)=>{if(o>t-6&&o<=t){s+=z.p[i];pk=Math.max(pk,z.p[i]);cp=Math.max(cp,z.c[i])}});
    const [l,cb]=lvl(s,pk,cp);if(l>=2)hi++;
    L.circle([z.lat,z.lon],{radius:D.km*1000,color:C[l],weight:2,fillColor:C[l],fillOpacity:.3})
      .bindPopup('<b>'+esc(z.name)+'</b><br>'+N[l]+'<br>6 h rain to '+clock(t)+': '+s.toFixed(1)+' mm<br>Peak: '+pk.toFixed(1)+' mm/h').addTo(zl);
    if(cb){cbs.push(z.name);L.marker([z.lat,z.lon],{icon:L.divIcon({className:'',html:'<div class="police"></div>',iconSize:[22,22]})})
      .bindPopup('<b>Cloudburst signal: '+esc(z.name)+'</b><br>Model-based, not an official warning').addTo(zl)}
  });
  D.inc.forEach(e=>{
    const age=t+e.h;if(age<0)return;shown++;
    L.circleMarker([e.lat,e.lon],{radius:age<=1?9:6,color:'#000',weight:1,fillColor:age<=1?'#d32f2f':age<=6?'#f28c28':'#888',fillOpacity:.9,className:age<=1?'pulse':''})
      .bindPopup('<b>Landslide: '+esc(e.place)+'</b><br>'+esc(e.time)+' IST<br>'+esc(e.src)).addTo(il)});
  document.getElementById('lab').textContent=lab(t);
  document.getElementById('sub').textContent=' \\u00b7 '+hi+' high/severe zones \\u00b7 '+shown+' landslides shown';
  const b=document.getElementById('cb');b.style.display=cbs.length?'block':'none';
  b.textContent='CLOUDBURST SIGNAL: '+cbs.join(', ')+(D.demo?' (SIMULATED)':'');
}
document.getElementById('s').oninput=e=>draw(Number(e.target.value));
draw(0);
</script></body></html>"""


def render_map(zones, incidents, demo):
    data = dict(
        zones=[{k: z[k] for k in ("name", "lat", "lon", "offs", "p", "c")} for z in zones],
        inc=[dict(place=i["place"], lat=i["lat"], lon=i["lon"], src=i["source"],
                  time=i["t"].strftime("%d %b %I:%M %p"),
                  h=(datetime.now(IST) - i["t"]).total_seconds() / 3600) for i in incidents],
        km=ZONE_KM, demo=demo, now=int(datetime.now(UTC).timestamp() * 1000),
    )
    payload = json.dumps(data).replace("</", "<\\/")
    html = MAP_HTML.replace("__DATA__", payload)
    if hasattr(st, "iframe"):
        st.iframe(html, height=640)
    else:
        components.html(html, height=640, scrolling=False)


@st.cache_data(ttl=1800)
def fetch_stays(lat, lon, radius_m):
    q = (f'[out:json][timeout:12];(nwr["tourism"~"^(hotel|guest_house|hostel|resort)$"](around:{radius_m},{lat},{lon});'
         f'nwr["amenity"="restaurant"](around:{radius_m},{lat},{lon}););out center 120;')
    encoded_query = ("data=" + quote(q)).encode()
    js = None
    for endpoint in (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ):
        js = _get_json(endpoint, data=encoded_query, timeout=15)
        if js is not None:
            break
    if js is None:
        return None
    rows = []
    for e in js.get("elements", []):
        t = e.get("tags", {})
        la = e.get("lat") or e.get("center", {}).get("lat")
        lo = e.get("lon") or e.get("center", {}).get("lon")
        if not t.get("name") or la is None or lo is None:
            continue
        kind = "Restaurant" if t.get("amenity") == "restaurant" else t.get("tourism", "stay").replace("_", " ").title()
        rows.append(dict(name=t["name"], kind=kind, lat=la, lon=lo, phone=t.get("phone") or t.get("contact:phone", "")))
    return rows


def render_stays(lat, lon, zones):
    st.subheader("Nearest hotels & restaurants")
    c1, c2 = st.columns(2)
    radius = c1.slider("Search radius (km)", 3, 30, 10)
    show = c2.radio("Show", ["Hotels / stays", "Restaurants", "Both"], horizontal=True)
    rows = fetch_stays(round(lat, 3), round(lon, 3), radius * 1000)
    if rows is None:
        st.warning("Nearby places are temporarily unavailable from OpenStreetMap's data service.")
        if st.button("Retry nearby places", key="retry_stays_lookup"):
            fetch_stays.clear()
            st.rerun()
        return
    danger = [z for z in zones if z["level"] >= 2 or z["cb"]]
    out = []
    for r in rows:
        if show == "Hotels / stays" and r["kind"] == "Restaurant":
            continue
        if show == "Restaurants" and r["kind"] != "Restaurant":
            continue
        inside = [z["name"] for z in danger if _km(r["lat"], r["lon"], z["lat"], z["lon"]) <= ZONE_KM]
        out.append({
            "Name": r["name"], "Type": r["kind"], "Distance (km)": round(_km(lat, lon, r["lat"], r["lon"]), 1),
            "Alert status": "Inside alert zone: " + inside[0] if inside else "Outside current alert zones",
            "Phone": r["phone"],
            "Directions": f"https://www.google.com/maps/dir/?api=1&destination={r['lat']},{r['lon']}",
        })
    if not out:
        st.info("Nothing found in this radius. Try a larger radius.")
        return
    df = pd.DataFrame(out).sort_values(
        ["Alert status", "Distance (km)"],
        key=lambda values: (~values.str.startswith("Inside") if values.name == "Alert status" else values),
    )
    st.dataframe(df.head(40), hide_index=True, use_container_width=True,
                 column_config={"Directions": st.column_config.LinkColumn("Directions", display_text="Open map")})
    st.caption("Source: OpenStreetMap (can be incomplete). 'Outside alert zones' only means outside the circles above. "
               "It does not prove a place is safe or open. Call ahead and follow official advice.")


def render_hazard_intel(lat, lon):
    st.subheader("Alert zones: past, now and next 36 hours")
    demo = st.toggle("Demo mode (simulated landslides + simulated cloudburst at Nahan)", value=False)
    raw = fetch_forecast()
    if raw is None:
        st.warning("Weather forecast is unavailable right now, so alert zones can't be drawn.")
    zones = build_zones(raw, demo)
    if raw is not None and not zones:
        st.warning("The forecast response contained no usable hotspot data; alert zones are unavailable.")
    elif zones and len(zones) < len(HOTSPOTS):
        st.info(f"Forecast coverage is partial: {len(zones)} of {len(HOTSPOTS)} hotspot zones are available.")
    incidents = load_incidents(demo)

    cb = [z["name"] for z in zones if z["cb"]]
    if cb:
        police_banner(cb, demo)
    for line in briefing(zones, incidents):
        st.markdown("- " + line)
    render_map(zones, incidents, demo)
    st.caption("Zones are 15 km circles coloured from Open-Meteo rainfall (past hours = model analysis, future = forecast). "
               "Cloudburst signal = model rain of 50+ mm/h, or 30+ mm/h with high CAPE; it is a warning sign, not a confirmation. "
               "Landslide markers come only from the incident log, not from a live national feed.")

    with st.expander("Log a landslide report"):
        with st.form("mw_log", clear_on_submit=True):
            place = st.text_input("Place (e.g. Chamoli highway)")
            c1, c2, c3 = st.columns(3)
            la = c1.number_input("Latitude", value=30.30, format="%.4f")
            lo = c2.number_input("Longitude", value=78.60, format="%.4f")
            tm = c3.time_input("Time (IST)", value=datetime.now(IST).time())
            src = st.text_input("Source (official / news link)")
            if st.form_submit_button("Add to map") and place:
                t = datetime.combine(datetime.now(IST).date(), tm, tzinfo=IST)
                if t > datetime.now(IST):
                    t -= timedelta(days=1)
                st.session_state.setdefault("mw_incidents", []).append(
                    dict(place=place, lat=la, lon=lo, t=t, source=src or "User report"))
                st.rerun()
        st.caption("Or add rows to incidents.csv next to app.py with columns: time_ist,place,lat,lon,source")

    render_stays(lat, lon, zones)
