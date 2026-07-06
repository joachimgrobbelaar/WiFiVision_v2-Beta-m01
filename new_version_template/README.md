# WiFiVision v2.0 — New Version Template (Real Hardware / m01-v1)

> **Template snapshot of the upgraded WiFiVision v2 Beta (m01) pipeline with real Wi-Fi hardware CSI data ingestion, bistatic ranging, 2D Gaussian splatting, and live Three.js WebGL dashboard.**

This folder contains the latest production version of the WiFiVision v2 pipeline.

## What is new in this version

| Feature | Description |
|---------|-------------|
| **Real Hardware JSONL Ingestion** | Automatically discovers and loads real CSI captures (`*.csi.jsonl`) from the `RuView/data/recordings/` directory |
| **Bistatic Ranging & Relative Positioning** | Calculates real-time fused distance between the Router (TX) and Client (RX) using ToF ($d = \tau \cdot c$), AoA angle, and RSSI |
| **Anisotropic 2D Gaussian Splatting** | Each RF reflection point grows as a 2D Gaussian splat on a pixel grid; expansion halts immediately upon colliding with an adjacent splat wavefront (boundary collision) |
| **Spatial Edge Detection** | Applies 2D Sobel gradient operators across the splatted surface field to delineate wall and obstacle boundary edges |
| **Three.js WebGL + Chart.js Dashboard** | Animated, self-contained 3D room scene with live RF pulse wavefront rings, Gaussian splat overlays, Doppler walking targets, vital sign subjects, real-time telemetry graphs, and interactive feature toggles |
| **Real Hardware Device Telemetry** | Receiver spec card in the dashboard shows actual capture metadata: frame count, sample rate, RSSI, hardware file name, ranging distance |

## What this version includes

| File | Description |
|------|-------------|
| `simulation.py` | Full end-to-end pipeline: real CSI ingestion → DSP → 2D-MUSIC → bistatic ranging → Gaussian splatting → edge detection → 3D dashboard |
| `ingestion.py` | Upgraded ingestion with `load_from_jsonl` / `load_capture_file` and auto-discovery logic |
| `dsp_engine.py` | DSP engine with SFO/CFO, 2D-MUSIC, material classifier, vital sign extraction |
| `geometry_mapping.py` | 2D & 3D bistatic geometry, DBSCAN clustering, anisotropic Gaussian splatting with boundary collision |
| `main_cli.py` | Unified CLI entry-point (`--test`, `--diag`, `--all`) |
| `3d_room_geometry.html` | **Live Three.js WebGL + Chart.js** interactive 3D HTML dashboard |
| `3d_room_geometry_map.png` | 3D bistatic spatial map with Router, Client, and reconstructed obstacles |
| `room_geometry_reconstruction.png` | 2D-MUSIC wall mapping with Gaussian splat fields and Sobel edge detection |
| `vital_sign_dashboard.png` | Doppler tracking, respiration waveform, and heart-rate FFT spectrum |
| `env_check.sh` | RF hardware diagnostics shell script |
| `run_pipeline.sh` / `.bat` | Linux & Windows launcher scripts |
| `README_HOW_IT_WORKS.md` | Full system architecture and RF physics reference guide |

## Pipeline output (real hardware capture run)

```
[+] LOADED REAL HARDWARE CSI: 2500 frames, sample rate 34.4 Hz, mean RSSI 1.0 dBm
[+] Router ↔ Laptop Fused Ranging Distance: 0.38 m (AoA: 1.0°, ToF: 1.00 ns)
[+] Relative Cartesian Position: X=0.38m, Y=0.01m
[+] Surface splatting complete: 3896 boundary collision pixels halted.
[+] Extracted Vital Signs: Respiration = 9.6 BPM, Heart Rate = 60.0 BPM
[+] Phase 5 Simulation & Dashboard Pipeline completed successfully!
```

## How to run this version

```bash
# Option A — Standalone Linux executable (no Python install required)
./dist/WifiVision_CSI_Pipeline

# Option B — Python virtual environment
./.venv/bin/python simulation.py

# Run unit tests
./dist/WifiVision_CSI_Pipeline --test

# Rebuild the standalone binary
./.venv/bin/pyinstaller WifiVision_CSI_Pipeline.spec --clean -y
```

---

_The original synthetic-simulation-only version is preserved in the `old_version_template/` folder._
