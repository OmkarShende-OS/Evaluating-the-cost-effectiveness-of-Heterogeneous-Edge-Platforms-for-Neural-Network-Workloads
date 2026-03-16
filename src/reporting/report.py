"""Markdown report generation helpers for benchmark outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from src.metrics.edcp import EDCPRecord


def write_edcp_markdown(records: Iterable[EDCPRecord], output_path: Path) -> None:
    lines = [
        "# EDCP Comparison Report",
        "",
        "| Device | Runtime | EDCP | Normalized EDCP |",
        "|---|---|---:|---:|",
    ]
    for r in records:
        norm = f"{r.edcp_normalized:.4f}" if r.edcp_normalized is not None else "-"
        lines.append(f"| {r.device} | {r.runtime} | {r.edcp:.6f} | {norm} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
