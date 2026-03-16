# Contributing

Thanks for helping improve this edge benchmarking toolkit.

## Contribution scope

- New runtime adapters
- Additional device profiles
- Improved metrics/reporting
- Reproducibility and documentation enhancements

## Development guidelines

- Keep scripts CLI-driven (`argparse`) and path-agnostic.
- Do not hardcode machine-specific paths.
- Write outputs as structured CSV/JSON where possible.
- Add/update tests for new metrics and reporting logic.

## Pull request checklist

- [ ] Changes are documented
- [ ] CLI help text is clear
- [ ] Tests pass locally
- [ ] No private data or credentials included
