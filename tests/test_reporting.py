from pathlib import Path
from src.metrics.edcp import EDCPRecord
from src.reporting.report import write_edcp_markdown


def test_report_writer(tmp_path: Path):
    out = tmp_path / "report.md"
    rows = [
        EDCPRecord(
            device="d1",
            runtime="r1",
            energy_j=0.1,
            delay_s=0.01,
            cost_usd=100,
            edcp=0.1,
            edcp_normalized=1.0,
        )
    ]
    write_edcp_markdown(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "EDCP Comparison Report" in text
