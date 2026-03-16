# Metrics

## Core metrics

- **Inference latency** (ms): kernel/runtime execution time.
- **Total latency** (ms): end-to-end path including pre/post operations.
- **Throughput** (FPS): images processed per second.
- **Energy per inference** (J): per-sample energy estimate.
- **Average power** (W): platform power draw over benchmark window.
- **Temperature** (°C): thermal behavior under sustained load.
- **Cost** (USD): platform acquisition cost used for comparative analysis.
- **Accuracy** (optional): model quality context for trade-offs.

## Derived metrics

- **EDP** = Energy × Delay
- **ED2P** = Energy × Delay²
- **EDCP** = Energy × Delay × Cost

EDCP is emphasized in this repository as the deployment-cost-aware metric introduced in the paper context.
