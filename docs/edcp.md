# EDCP (Energy × Delay × Cost Product)

## Definition

$$
EDCP = E \times D \times C
$$

Where:

- $E$ = energy per inference (J)
- $D$ = delay per inference (s)
- $C$ = platform cost (USD)

Lower is better.

## Why EDCP

Single-axis comparisons (e.g., throughput-only) can over-favor expensive devices. EDCP captures deployment reality by combining efficiency, latency, and cost.

## Normalized EDCP

For cross-device ranking, we also compute:

$$
EDCP_{norm} = \frac{EDCP}{EDCP_{baseline}}
$$

This makes relative comparisons easier in tables and reports.

## Code references

- `src/metrics/edcp.py`
- `scripts/compute_edcp.py`
