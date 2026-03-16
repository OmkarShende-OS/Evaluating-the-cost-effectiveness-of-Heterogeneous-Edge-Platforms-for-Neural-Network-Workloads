import json
from pathlib import Path


def test_device_config_loads():
    path = Path("configs/devices/sample_devices.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "devices" in payload
    assert len(payload["devices"]) >= 1
