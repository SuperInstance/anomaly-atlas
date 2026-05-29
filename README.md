# anomaly-atlas

**Unified conservation-based anomaly detection across all domains — one pattern (Laplacian → CR → threshold) applied to music, finance, climate, social, protein, neural, and PX4 data.**

The core insight: anomalies violate the smooth structure captured by the graph Laplacian. Build a Laplacian, compute the conservation ratio, threshold on deviation. This atlas demonstrates the same detector architecture working across 7 domains with domain-specific graph construction.

## What This Gives You

- **Universal anomaly pattern** — Laplacian → conservation ratio → threshold
- **7 domains** — Music, Finance, Climate, Social, Protein, Neural, PX4
- **Domain-specific graphs** — transition matrices, correlation networks, contact maps
- **Conservation ratio as anomaly score** — low CR = anomalous
- **Visualization suite** — multi-panel domain comparison plots
- **Noise robustness** — tested across noise levels

## Quick Start

```bash
pip install numpy matplotlib
python anomaly_atlas.py
```

Outputs go to `figures/` with multi-panel PNG plots.

## How It Fits

Part of the SuperInstance ecosystem:

- **[conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python)** — Core SDK
- **[px4-conservation-poc](https://github.com/SuperInstance/px4-conservation-poc)** — PX4-specific version
- **anomaly-atlas** — Unified cross-domain atlas (this repo)

## License

MIT
