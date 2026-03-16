from src.profiling.power_thermal import summarize_power_temperature


def test_power_temperature_summary():
    out = summarize_power_temperature([10.0, 20.0], [50.0, 60.0])
    assert out["avg_power_w"] == 15.0
    assert out["avg_temperature_c"] == 55.0
    assert out["max_temperature_c"] == 60.0
