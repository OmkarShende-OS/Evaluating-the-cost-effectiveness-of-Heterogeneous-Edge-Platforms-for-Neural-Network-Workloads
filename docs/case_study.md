# Case Study: Edge Traffic Analytics Deployment

## Scenario

A smart-city traffic analytics node must run near-real-time vehicle inference under power and cost constraints.

## Decision process

1. Benchmark candidate devices/runtimes for target model.
2. Compare latency, energy, and thermal behavior.
3. Compute EDCP to incorporate hardware cost.
4. Choose deployment mode:
   - accuracy-priority runtime for low traffic periods
   - speed/efficiency runtime for high traffic periods

## Outcome

This workflow turns benchmark outputs into deployment decisions rather than isolated performance charts.
