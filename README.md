# Shock-and-Awe

Minimal extracted workspace for storm-shock normal analysis.

## Contents
- `CDASWS Downloader.ipynb`
- `Plots.ipynb`
- `Normals Solver.nb` and `Normals Numerics.nb`
- `Data/` dated parquet folders
- `Shocks/` JSON inputs/outputs
- `requirements.txt`

## Setup
Use the `icme3.12-metal` environment (or install from `requirements.txt`).

## Tests
Keep all tests under the `Tests/` directory. Run the offline cadence
checks from the repository root:

```bash
conda run -n icme3.12-metal python -m unittest discover -s Tests -p test_downloader_cadence.py -v
```

The network integration test is opt-in; see
`Tests/integration/test_downloader_wolfram_mx3.py` for its invocation.
