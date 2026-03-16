from src.metrics.edcp import compute_edcp


def test_compute_edcp_positive_values():
    assert compute_edcp(0.1, 0.02, 100.0) == 0.2
