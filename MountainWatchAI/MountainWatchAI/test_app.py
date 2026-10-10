from app import (
    calculate_risk_score,
    classify_rainfall_signal,
    classify_risk,
    fetch_recent_earthquakes,
    normalize_live_weather_to_inputs,
)
from hazard_intel import fetch_stays, level_for, rain_window_stats


def test_calculate_risk_score():
    score = calculate_risk_score(120, 300, 45, 70)
    assert score == 71


def test_normalize_live_weather_to_inputs():
    payload = {
        "rain_24h": 115,
        "rain_3d": 350,
        "soil_moisture": 68,
    }
    inputs = normalize_live_weather_to_inputs(payload)
    assert inputs == {
        "rain_24h": 115,
        "rain_3d": 350,
        "soil_moisture": 68,
    }


def test_fetch_recent_earthquakes_handles_empty_results(monkeypatch):
    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            return b'{"features": []}'

    monkeypatch.setattr("app.urlopen", lambda *args, **kwargs: DummyResponse())
    df = fetch_recent_earthquakes(30.3, 78.0, 250)
    assert list(df.columns) == ["place", "mag", "time", "latitude", "longitude"]
    assert df.empty


def test_classify_risk():
    assert classify_risk(25) == "Low"
    assert classify_risk(40) == "Moderate"
    assert classify_risk(66) == "Elevated"
    assert classify_risk(82) == "High"


def test_classify_rainfall_signal():
    assert classify_rainfall_signal(0) == "No heavy rain signal"
    assert classify_rainfall_signal(20) == "Heavy rain signal"
    assert classify_rainfall_signal(50) == "Extreme hourly rain signal"


def test_cloudburst_uses_cape_from_peak_rain_hour():
    separated_rain_and_cape = rain_window_stats([35, 0], [0, 1800])
    coincident_rain_and_cape = rain_window_stats([35, 0], [1800, 0])

    assert level_for(*separated_rain_and_cape) == (2, False)
    assert level_for(*coincident_rain_and_cape) == (3, True)


def test_fetch_stays_falls_back_to_backup_overpass(monkeypatch):
    requested = []

    def fake_get_json(url, data=None, timeout=25):
        requested.append(url)
        return None if len(requested) == 1 else {"elements": []}

    monkeypatch.setattr("hazard_intel._get_json", fake_get_json)
    fetch_stays.clear()

    assert fetch_stays(30.3165, 78.0322, 10000) == []
    assert requested == [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
