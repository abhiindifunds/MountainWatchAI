from app import calculate_risk_score, classify_risk, fetch_recent_earthquakes, normalize_live_weather_to_inputs


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
