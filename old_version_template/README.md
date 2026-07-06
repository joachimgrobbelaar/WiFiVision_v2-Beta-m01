# WiFiVision v2.0 — Old Version Template (Initial Release / m01-v0)

> **Template snapshot of the original WiFiVision v2 Beta (m01) pipeline as released in the initial commit.**

This folder preserves the first published version of the WiFiVision v2 pipeline exactly as it was shipped. It is kept here as a reference template so that future versions can be compared against, forked from, or rolled back to.

## What this version includes

| File | Description |
|------|-------------|
| `simulation.py` | Original room-simulation pipeline using **synthetic CSI data**; no real hardware data ingestion |
| `ingestion.py` | Base ingestion engine without JSONL hardware-capture support |
| `dsp_engine.py` | Core DSP: SFO/CFO sanitisation, 2D-MUSIC, material classifier |
| `geometry_mapping.py` | 2D & 3D bistatic geometry mapping, DBSCAN clustering |
| `main_cli.py` | Unified CLI entry-point (`--test`, `--diag`, `--all`) |
| `3d_room_geometry.html` | Static 3D Plotly HTML dashboard |
| `*.png` | Visual artifacts generated from the synthetic simulation run |
| `env_check.sh` | RF hardware diagnostics shell script |
| `run_pipeline.sh` / `.bat` | Linux & Windows launcher scripts |

## How to run this version

```bash
# 1. Create and activate the virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install numpy scipy matplotlib scikit-learn pyinstaller

# 3. Run the synthetic simulation
python simulation.py

# 4. Run unit tests
python main_cli.py --test
```

---

_For the latest version with real hardware CSI ingestion, bistatic ranging, 2D Gaussian splatting, and the live Three.js WebGL dashboard, see the `new_version_template/` folder or the root of this repository._
