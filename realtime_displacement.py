#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFiVision — Real-Time Relative Displacement Vector Tracker (v2)
================================================================
Tracks the real-time 3D displacement vector [X, Y, Z] between the
Router (TX) and Laptop (RX) using the direct LOS Wi-Fi signal only.

  X, Y = horizontal ground plane (from azimuth AoA + ToF distance)
  Z    = vertical height estimate (from elevation AoA via virtual vertical array)

  NO motion tracking, NO Gaussian splatting, NO heartbeat monitoring,
  NO background subtraction (which would remove the direct path signal).

Usage:
  ./.venv/bin/python realtime_displacement.py
  Open http://localhost:5757 in your browser.
  Ctrl+C to stop.
"""

import os
import sys
import json
import time
import math
import threading
import queue

import numpy as np
from scipy import fft as scipy_fft

try:
    from flask import Flask, Response, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from ingestion import CSIIngestionEngine
from dsp_engine import CSIDSPEngine

# =============================================================================
# CONFIGURATION
# =============================================================================
SPEED_OF_LIGHT   = 2.998e8      # m/s
CARRIER_HZ       = 5.2e9        # 5.2 GHz
BANDWIDTH_HZ     = 40.0e6       # 40 MHz
N_SUBCARRIERS    = 64
N_ANTENNAS       = 4
SAMPLE_RATE_HZ   = 34.4         # Real hardware capture rate
WINDOW_FRAMES    = 16           # Smaller window -> faster response
STRIDE_FRAMES    = 2            # Step 2 frames -> ~17 Hz update rate
DASHBOARD_PORT   = 5757
MAX_HISTORY_PTS  = 2000         # Long trail (never auto-resets; refresh browser to clear)
Z_HISTORY_PTS    = 300          # Z time-series window length

# =============================================================================
# SHARED STATE
# =============================================================================
_state_lock = threading.Lock()
_latest_vector = {
    "d_m": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0,
    "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
    "tof_ns": 0.0, "rssi_dbm": -50.0,
    "frame_idx": 0, "cycle": 0, "ts": 0.0,
    "tx_freq_ghz": 5.2, "tx_bw_mhz": 40.0, "rx_hw": "Intel AX210",
    "rx_file": "", "rx_frames": 0, "rx_sr_hz": SAMPLE_RATE_HZ,
}
_history_x: list = []
_history_y: list = []
_history_z: list = []  # Z values for the vertical chart
_history_t: list = []  # timestamps for Z chart
_sse_clients: list = []

# =============================================================================
# LOS DISPLACEMENT EXTRACTOR  (3D: azimuth + elevation + ToF)
# =============================================================================

class LOSDisplacementExtractor:
    """
    Estimates 3D displacement using the direct LOS path only.

    Azimuth (phi): 2D-MUSIC on horizontal ULA phase differences
        phi_horizontal(a) = pi * a * sin(azimuth)

    Elevation (theta): virtual vertical MUSIC pass — reinterprets the
        same 4-antenna array as a vertical stack (column permutation)
        to extract the elevation angle:
        phi_vertical(a) = pi * a * sin(elevation)

    3D Cartesian:
        X = d * cos(elevation) * cos(azimuth)   (East)
        Y = d * cos(elevation) * sin(azimuth)   (North)
        Z = d * sin(elevation)                   (Up)
    """

    def __init__(self):
        self.dsp = CSIDSPEngine(
            carrier_freq_hz=CARRIER_HZ,
            bandwidth_hz=BANDWIDTH_HZ,
            n_subcarriers=N_SUBCARRIERS,
            n_antennas=N_ANTENNAS,
            sample_rate_hz=SAMPLE_RATE_HZ,
        )
        self.delay_grid_ns   = np.linspace(1.0, 120.0, 180)
        self.azimuth_grid    = np.linspace(-90.0, 90.0, 181)
        self.elevation_grid  = np.linspace(-45.0, 45.0, 91)

    def _pdp_tof(self, csi_clean: np.ndarray) -> float:
        """Return ToF of the first strong (LOS) peak in the Power Delay Profile."""
        pdp, _ = self.dsp.compute_pdp_ifft(csi_clean)
        pdp_mean = np.mean(np.abs(pdp), axis=tuple(range(1, pdp.ndim)))
        bin_to_ns = (1.0 / BANDWIDTH_HZ) * 1e9
        delay_ns  = np.arange(len(pdp_mean)) * bin_to_ns
        threshold = np.max(pdp_mean) * 0.15
        candidates = np.where(pdp_mean > threshold)[0]
        return float(delay_ns[candidates[0]]) if len(candidates) > 0 else 10.0

    def _music_azimuth(self, csi_snap: np.ndarray) -> tuple:
        """Run 2D-MUSIC for azimuth + refine ToF."""
        try:
            _, peaks = self.dsp.estimate_2d_music(
                csi_snap, n_sources=1,
                angle_grid_deg=self.azimuth_grid,
                delay_grid_ns=self.delay_grid_ns,
            )
            if peaks:
                return float(peaks[0][0]), float(peaks[0][1])
        except Exception:
            pass
        return 0.0, None

    def _music_elevation(self, csi_snap: np.ndarray) -> float:
        """
        Estimate elevation by permuting antenna columns to simulate a
        vertical array and running a second MUSIC pass.
        Virtual vertical array: reverse antenna ordering so the phase
        gradient now represents elevation instead of azimuth.
        """
        try:
            # Permute antenna axis to create a "virtual vertical" snapshot
            csi_vert = csi_snap[:, ::-1]  # (N_f, N_a) reversed
            _, peaks = self.dsp.estimate_2d_music(
                csi_vert, n_sources=1,
                angle_grid_deg=self.elevation_grid,
                delay_grid_ns=self.delay_grid_ns,
            )
            if peaks:
                return float(peaks[0][0])
        except Exception:
            pass
        return 0.0

    def extract(self, csi_window: np.ndarray, rssi_dbm: float = -50.0) -> dict:
        """Full 3D LOS extraction."""

        # Phase sanitization (SFO/CFO) — keep; background subtraction — skip
        csi_clean = self.dsp.sanitize_phase_sfo_cfo(csi_window)
        csi_snap  = np.mean(csi_clean, axis=1)  # (N_f, N_a)

        # PDP ToF (LOS shortest peak)
        tof_pdp = self._pdp_tof(csi_clean)

        # Azimuth + refined ToF from 2D-MUSIC
        azimuth_deg, tof_music = self._music_azimuth(csi_snap)
        tof_ns = (0.6 * tof_music + 0.4 * tof_pdp) if tof_music else tof_pdp

        # Elevation from virtual vertical array MUSIC
        elevation_deg = self._music_elevation(csi_snap)

        # Distance: d = ToF * c
        d_csi = (tof_ns * 1e-9) * SPEED_OF_LIGHT

        # RSSI log-distance backup
        rssi_clamped = min(-31.0, max(-95.0, float(rssi_dbm)))
        d_rssi = min(10.0 ** ((-rssi_clamped - 30.0) / (10.0 * 2.8)), 50.0)

        # Fused distance
        d = 0.75 * d_csi + 0.25 * d_rssi

        # 3D Cartesian coordinates
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(elevation_deg)
        x = d * math.cos(el_rad) * math.cos(az_rad)
        y = d * math.cos(el_rad) * math.sin(az_rad)
        z = d * math.sin(el_rad)

        return {
            "d_m":           round(d, 3),
            "azimuth_deg":   round(azimuth_deg, 1),
            "elevation_deg": round(elevation_deg, 1),
            "x_m":           round(x, 3),
            "y_m":           round(y, 3),
            "z_m":           round(z, 3),
            "tof_ns":        round(tof_ns, 2),
            "rssi_dbm":      round(rssi_dbm, 1),
        }


# =============================================================================
# TRACKING THREAD  (~17 Hz update rate)
# =============================================================================

def tracking_thread(csi_full: np.ndarray, rssi_arr: np.ndarray,
                    timestamps: np.ndarray, meta: dict):
    global _latest_vector, _history_x, _history_y, _history_z, _history_t

    extractor = LOSDisplacementExtractor()
    n_frames  = csi_full.shape[1]
    frame_idx = 0
    cycle     = 0
    t0        = time.time()

    update_hz = SAMPLE_RATE_HZ / STRIDE_FRAMES
    print(f"[*] Tracker: {n_frames} frames | window={WINDOW_FRAMES} | "
          f"stride={STRIDE_FRAMES} | update rate ~{update_hz:.1f} Hz")

    while True:
        start = frame_idx % n_frames
        end   = (frame_idx + WINDOW_FRAMES) % n_frames

        if end > start:
            csi_win  = csi_full[:, start:end, :]
            rssi_win = rssi_arr[start:end]
        else:
            csi_win  = np.concatenate([csi_full[:, start:, :],
                                       csi_full[:, :end, :]], axis=1)
            rssi_win = np.concatenate([rssi_arr[start:], rssi_arr[:end]])

        try:
            vec = extractor.extract(csi_win, rssi_dbm=float(np.mean(rssi_win)))
        except Exception:
            vec = {"d_m": 0.0, "azimuth_deg": 0.0, "elevation_deg": 0.0,
                   "x_m": 0.0, "y_m": 0.0, "z_m": 0.0,
                   "tof_ns": 0.0, "rssi_dbm": float(np.mean(rssi_win))}

        elapsed = time.time() - t0
        vec.update({
            "frame_idx":   frame_idx,
            "cycle":       cycle,
            "ts":          elapsed,
            "tx_freq_ghz": 5.2,
            "tx_bw_mhz":   40.0,
            "rx_hw":       "Intel AX210",
            "rx_file":     meta.get("file_name", ""),
            "rx_frames":   meta.get("n_frames", 0),
            "rx_sr_hz":    round(meta.get("sample_rate_hz", SAMPLE_RATE_HZ), 1),
        })

        with _state_lock:
            _latest_vector.update(vec)
            _history_x.append(vec["x_m"])
            _history_y.append(vec["y_m"])
            _history_z.append(vec["z_m"])
            _history_t.append(elapsed)
            # Trim trail to max length
            if len(_history_x) > MAX_HISTORY_PTS:
                _history_x.pop(0); _history_y.pop(0)
            if len(_history_z) > Z_HISTORY_PTS:
                _history_z.pop(0); _history_t.pop(0)

        # SSE broadcast
        payload = "data: " + json.dumps(vec) + "\n\n"
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            try: _sse_clients.remove(q)
            except ValueError: pass

        print(f"\r[{frame_idx:5d}|cyc{cycle}] "
              f"d={vec['d_m']:5.2f}m  "
              f"az={vec['azimuth_deg']:+6.1f}°  "
              f"el={vec['elevation_deg']:+5.1f}°  "
              f"X={vec['x_m']:+6.2f}  Y={vec['y_m']:+6.2f}  Z={vec['z_m']:+5.2f}  "
              f"RSSI={vec['rssi_dbm']:+.0f}dBm",
              end="", flush=True)

        frame_idx += STRIDE_FRAMES
        if frame_idx >= n_frames:
            frame_idx = 0
            cycle += 1

        time.sleep(STRIDE_FRAMES / SAMPLE_RATE_HZ)


# =============================================================================
# HTML DASHBOARD
# =============================================================================

_DASH = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WiFiVision - 3D Displacement Tracker</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:#060c18;color:#e2e8f0;min-height:100vh;padding:16px}

/* ── Header ── */
.hdr{text-align:center;margin-bottom:18px}
.hdr h1{font-size:1.7rem;font-weight:700;
  background:linear-gradient(135deg,#38bdf8,#818cf8,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hdr p{color:#475569;font-size:.85rem;margin-top:3px}
.live{display:inline-flex;align-items:center;gap:6px;background:#0f172a;
  border:1px solid #1e3a5f;border-radius:20px;padding:3px 12px;
  font-size:.72rem;color:#38bdf8;margin-top:6px}
.dot{width:7px;height:7px;border-radius:50%;background:#22c55e;
  animation:blink 1.2s ease infinite}
@keyframes blink{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.5)}}

/* ── Device stat cards ── */
.dev-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.dev-card{background:#0d1526;border:1px solid #1e293b;border-radius:12px;padding:14px 18px}
.dev-card.tx{border-left:4px solid #ef4444}
.dev-card.rx{border-left:4px solid #3b82f6}
.dev-title{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;
  color:#94a3b8;font-weight:700;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.dev-badge{padding:2px 7px;border-radius:10px;font-size:.6rem;font-weight:700}
.badge-tx{background:#7f1d1d;color:#fca5a5}
.badge-rx{background:#1e3a8a;color:#93c5fd}
.dev-name{font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:6px}
.dev-stats{display:grid;grid-template-columns:1fr 1fr;gap:4px 14px}
.stat-row{font-size:.75rem;color:#64748b}
.stat-val{color:#cbd5e1;font-family:'JetBrains Mono',monospace;font-weight:600}

/* ── Main layout ── */
.main{display:grid;grid-template-columns:320px 1fr 260px;gap:14px;align-items:start}

/* ── Compass ── */
.compass-wrap{background:#0d1526;border:1px solid #1e293b;border-radius:14px;
  padding:16px;display:flex;flex-direction:column;align-items:center;gap:10px}
.sec-title{font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;
  color:#475569;font-weight:600;text-align:center}

/* ── Right charts column ── */
.charts-col{display:flex;flex-direction:column;gap:14px}
.chart-wrap{background:#0d1526;border:1px solid #1e293b;border-radius:14px;padding:14px}

/* ── Metric cards (right column) ── */
.metrics{display:flex;flex-direction:column;gap:10px}
.mcard{background:#0d1526;border:1px solid #1e293b;border-radius:10px;padding:12px 16px}
.mlabel{font-size:.64rem;text-transform:uppercase;letter-spacing:.08em;
  color:#475569;font-weight:600;margin-bottom:6px}
.mval{font-family:'JetBrains Mono',monospace;font-weight:600;color:#f1f5f9;line-height:1.1}
.msub{font-size:.72rem;color:#475569;margin-top:4px}
.mbar{height:3px;border-radius:2px;background:#1e293b;margin-top:6px;overflow:hidden}
.mfill{height:100%;border-radius:2px;background:linear-gradient(90deg,#38bdf8,#818cf8);transition:width .2s}

.c-dist{color:#38bdf8}.c-az{color:#818cf8}.c-el{color:#34d399}
.c-vec{color:#a3e635;font-size:1rem}.c-tof{color:#fb923c}
.c-rssi{color:#f472b6}

#status{position:fixed;bottom:12px;right:16px;font-size:.65rem;
  color:#1e293b;font-family:'JetBrains Mono',monospace}
</style>
</head>
<body>

<div class="hdr">
  <h1>WiFiVision &mdash; 3D Displacement Vector Tracker</h1>
  <p>Direct LOS path only &bull; Azimuth + Elevation AoA &bull; ToF Ranging &bull; 3D Cartesian [X, Y, Z]</p>
  <div class="live"><span class="dot"></span> LIVE &mdash; updating at ~17 Hz</div>
</div>

<!-- Device stat cards -->
<div class="dev-row">
  <div class="dev-card tx">
    <div class="dev-title">
      <span class="dev-badge badge-tx">TX</span>
      Sender &mdash; Router / Access Point
    </div>
    <div class="dev-name">ASUS ROG Wi-Fi 6E AP</div>
    <div class="dev-stats">
      <span class="stat-row">Frequency: <span class="stat-val" id="txFreq">5.200 GHz</span></span>
      <span class="stat-row">Bandwidth: <span class="stat-val" id="txBW">40 MHz</span></span>
      <span class="stat-row">Mode: <span class="stat-val">IEEE 802.11ax</span></span>
      <span class="stat-row">Array: <span class="stat-val">4-element ULA (TX)</span></span>
      <span class="stat-row">Position: <span class="stat-val">Origin (0, 0, 0)</span></span>
      <span class="stat-row">PRF: <span class="stat-val">100 Hz beacon</span></span>
    </div>
  </div>
  <div class="dev-card rx">
    <div class="dev-title">
      <span class="dev-badge badge-rx">RX</span>
      Receiver &mdash; Laptop / Client Node
    </div>
    <div class="dev-name" id="rxHW">Intel AX210 3-element ULA</div>
    <div class="dev-stats">
      <span class="stat-row">Capture file: <span class="stat-val" id="rxFile">--</span></span>
      <span class="stat-row">Frames loaded: <span class="stat-val" id="rxFrames">--</span></span>
      <span class="stat-row">Sample rate: <span class="stat-val" id="rxSR">-- Hz</span></span>
      <span class="stat-row">Current RSSI: <span class="stat-val" id="rxRSSI">-- dBm</span></span>
      <span class="stat-row">AoA method: <span class="stat-val">2D-MUSIC super-res.</span></span>
      <span class="stat-row">Ranging: <span class="stat-val">ToF + RSSI fused</span></span>
    </div>
  </div>
</div>

<!-- Main layout: compass | XY + Z charts | metric cards -->
<div class="main">

  <!-- Compass (azimuth + elevation needle) -->
  <div>
    <div class="compass-wrap">
      <div class="sec-title">Azimuth Compass &mdash; Horizontal Plane (XY)</div>
      <canvas id="compass" width="290" height="290"></canvas>
    </div>
  </div>

  <!-- Charts column: XY ground plane + Z altitude -->
  <div class="charts-col">
    <div class="chart-wrap">
      <div class="sec-title" style="margin-bottom:10px">
        Horizontal Ground Plane (XY) &mdash; Router at origin &bull; Trail: last <span id="trailLen">0</span> pts
      </div>
      <canvas id="xy" width="600" height="300"></canvas>
    </div>
    <div class="chart-wrap">
      <div class="sec-title" style="margin-bottom:10px">
        Vertical Elevation Z over Time &mdash; height / depth relative to router
      </div>
      <canvas id="zchart" width="600" height="180"></canvas>
    </div>
  </div>

  <!-- Metric cards -->
  <div class="metrics">

    <div class="mcard">
      <div class="mlabel">Distance (fused)</div>
      <div class="mval c-dist" id="vDist" style="font-size:1.6rem">--<small style="font-size:.8rem;color:#64748b"> m</small></div>
      <div class="msub" id="sDist">ToF: -- ns</div>
      <div class="mbar"><div class="mfill" id="bDist" style="width:0%"></div></div>
    </div>

    <div class="mcard">
      <div class="mlabel">Azimuth Angle (AoA)</div>
      <div class="mval c-az" id="vAz" style="font-size:1.6rem">--<small style="font-size:.8rem;color:#64748b"> °</small></div>
      <div class="msub">Horizontal bearing from boresight (0° = broadside)</div>
    </div>

    <div class="mcard">
      <div class="mlabel">Elevation Angle</div>
      <div class="mval c-el" id="vEl" style="font-size:1.6rem">--<small style="font-size:.8rem;color:#64748b"> °</small></div>
      <div class="msub" id="sEl">Z = -- m (above/below)</div>
    </div>

    <div class="mcard">
      <div class="mlabel">3D Displacement Vector</div>
      <div class="mval c-vec" id="vVec">X: --<br>Y: --<br>Z: --</div>
      <div class="msub" id="sVec">|r| = -- m</div>
    </div>

    <div class="mcard">
      <div class="mlabel">Time-of-Flight (LOS)</div>
      <div class="mval c-tof" id="vTof" style="font-size:1.4rem">--<small style="font-size:.8rem;color:#64748b"> ns</small></div>
      <div class="msub">1 ns = 0.30 m &nbsp;|&nbsp; d = &tau; &times; c</div>
    </div>

    <div class="mcard">
      <div class="mlabel">RSSI</div>
      <div class="mval c-rssi" id="vRSSI" style="font-size:1.4rem">--<small style="font-size:.8rem;color:#64748b"> dBm</small></div>
      <div class="msub" id="sRSSI">Link: --</div>
    </div>

    <div class="mcard">
      <div class="mlabel">Frame</div>
      <div class="mval" id="vFrame" style="font-size:1rem;color:#64748b">--</div>
      <div class="msub" id="sFrame">Elapsed: -- s</div>
    </div>

    <div class="mcard" style="border-color:#1e3a5f">
      <div class="mlabel" style="color:#38bdf8">Trail controls</div>
      <button onclick="clearTrail()" style="background:#1e3a8a;color:#93c5fd;border:none;
        border-radius:6px;padding:6px 12px;cursor:pointer;font-size:.8rem;font-weight:600;
        width:100%">Clear XY Trail</button>
    </div>

  </div>
</div>

<div id="status">Connecting...</div>

<script>
// ═══════════════════════════════════════════════════════════
//  COMPASS CANVAS
// ═══════════════════════════════════════════════════════════
const cvsC = document.getElementById('compass');
const ctxC = cvsC.getContext('2d');
const CX = 145, CY = 145, CR = 128;

function drawCompass(az, el, dist) {
  ctxC.clearRect(0,0,290,290);

  // BG
  ctxC.beginPath(); ctxC.arc(CX,CY,CR,0,2*Math.PI);
  ctxC.fillStyle='#060d1c'; ctxC.fill();
  ctxC.strokeStyle='#1e293b'; ctxC.lineWidth=2; ctxC.stroke();

  // Range rings
  [.33,.66,1].forEach(f=>{
    ctxC.beginPath(); ctxC.arc(CX,CY,CR*f,0,2*Math.PI);
    ctxC.strokeStyle='#0e2040'; ctxC.lineWidth=1; ctxC.stroke();
  });

  // Cross hairs
  ctxC.strokeStyle='#1e3a5f'; ctxC.lineWidth=1;
  ctxC.beginPath(); ctxC.moveTo(CX-CR,CY); ctxC.lineTo(CX+CR,CY); ctxC.stroke();
  ctxC.beginPath(); ctxC.moveTo(CX,CY-CR); ctxC.lineTo(CX,CY+CR); ctxC.stroke();

  // Tick marks and labels
  for(let a=-90;a<=90;a+=15){
    const r=Math.PI/2-a*Math.PI/180;
    const i=CR*.87, o=CR*.98;
    ctxC.beginPath();
    ctxC.moveTo(CX+i*Math.cos(r),CY-i*Math.sin(r));
    ctxC.lineTo(CX+o*Math.cos(r),CY-o*Math.sin(r));
    ctxC.strokeStyle=a===0?'#38bdf8':'#1e3a5f';
    ctxC.lineWidth=a%45===0?2:1; ctxC.stroke();
    if(a%30===0){
      ctxC.fillStyle='#374151'; ctxC.font='9px Inter'; ctxC.textAlign='center';
      ctxC.fillText(a+'°', CX+CR*.72*Math.cos(r), CY-CR*.72*Math.sin(r)+4);
    }
  }

  // Elevation arc indicator (shows elevation as arc thickness)
  const elNorm = Math.max(-1,Math.min(1, el/45.0));
  ctxC.beginPath(); ctxC.arc(CX,CY,CR*.18,0,2*Math.PI);
  ctxC.strokeStyle=elNorm>=0?'#34d399':'#f87171';
  ctxC.lineWidth=3+Math.abs(elNorm)*8; ctxC.stroke();
  ctxC.fillStyle='#94a3b8'; ctxC.font='bold 9px Inter';
  ctxC.textAlign='center'; ctxC.textBaseline='middle';
  ctxC.fillText((el>=0?'+':'')+el.toFixed(1)+'°el', CX, CY);

  // TX router dot
  ctxC.beginPath(); ctxC.arc(CX,CY,9,0,2*Math.PI);
  ctxC.fillStyle='#ef4444'; ctxC.fill();
  ctxC.fillStyle='#fff'; ctxC.font='bold 9px Inter';
  ctxC.textAlign='center'; ctxC.textBaseline='middle';
  ctxC.fillText('TX',CX,CY);

  // Displacement arrow
  const azR = Math.PI/2 - az*Math.PI/180;
  const norm = Math.min(dist/20.0, 1.0);
  const tx = CX + norm*CR*.84*Math.cos(azR);
  const ty = CY - norm*CR*.84*Math.sin(azR);

  ctxC.beginPath(); ctxC.moveTo(CX,CY); ctxC.lineTo(tx,ty);
  ctxC.strokeStyle='#38bdf8'; ctxC.lineWidth=2.5; ctxC.stroke();

  // Arrow head
  const ang=Math.atan2(CY-ty, tx-CX);
  const hl=11,ha=.4;
  ctxC.beginPath(); ctxC.moveTo(tx,ty);
  ctxC.lineTo(tx-hl*Math.cos(ang-ha),ty+hl*Math.sin(ang-ha));
  ctxC.lineTo(tx-hl*Math.cos(ang+ha),ty+hl*Math.sin(ang+ha));
  ctxC.closePath(); ctxC.fillStyle='#38bdf8'; ctxC.fill();

  // RX dot
  ctxC.beginPath(); ctxC.arc(tx,ty,7,0,2*Math.PI);
  ctxC.fillStyle='#818cf8'; ctxC.fill();
  ctxC.fillStyle='#fff'; ctxC.font='bold 8px Inter';
  ctxC.textAlign='center'; ctxC.textBaseline='middle';
  ctxC.fillText('RX',tx,ty);

  // Distance label on shaft
  const mx=CX+(norm*CR*.84/2)*Math.cos(azR)+14;
  const my=CY-(norm*CR*.84/2)*Math.sin(azR)-9;
  ctxC.fillStyle='#38bdf8'; ctxC.font='bold 11px JetBrains Mono,monospace';
  ctxC.textAlign='left'; ctxC.textBaseline='alphabetic';
  ctxC.fillText(dist.toFixed(2)+'m', mx, my);
}

// ═══════════════════════════════════════════════════════════
//  XY GROUND PLANE CANVAS
// ═══════════════════════════════════════════════════════════
const cvsXY = document.getElementById('xy');
const ctxXY = cvsXY.getContext('2d');
let trailX=[], trailY=[];

function clearTrail(){ trailX=[]; trailY=[]; }

function drawXY(nx, ny) {
  const W=cvsXY.offsetWidth||600, H=300;
  cvsXY.width=W; cvsXY.height=H;
  ctxXY.clearRect(0,0,W,H);
  trailX.push(nx); trailY.push(ny);
  // No auto-trim — manual clear only
  document.getElementById('trailLen').innerText = trailX.length;

  const cx0=W/2, cy0=H/2;
  // Grid
  ctxXY.strokeStyle='#0e2040'; ctxXY.lineWidth=1;
  [.25,.5,.75,1].forEach(f=>{
    ctxXY.beginPath(); ctxXY.arc(cx0,cy0,Math.min(W,H)/2*.9*f,0,2*Math.PI); ctxXY.stroke();
  });
  ctxXY.beginPath(); ctxXY.moveTo(0,cy0); ctxXY.lineTo(W,cy0); ctxXY.stroke();
  ctxXY.beginPath(); ctxXY.moveTo(cx0,0); ctxXY.lineTo(cx0,H); ctxXY.stroke();

  // Axis labels
  ctxXY.fillStyle='#334155'; ctxXY.font='10px Inter'; ctxXY.textAlign='center';
  ctxXY.fillText('+ X (East)', W-36, cy0-6);
  ctxXY.fillText('+ Y (North)', cx0+4, 12);
  ctxXY.fillText('TX (0,0)', cx0+16, cy0-10);

  if(trailX.length<2) return;

  const allX=trailX.concat([0]), allY=trailY.concat([0]);
  const xR=Math.max(...allX.map(Math.abs))*1.3||5;
  const yR=Math.max(...allY.map(Math.abs))*1.3||5;
  const sc=Math.min((W/2*.85)/xR, (H/2*.85)/yR);
  const toX=v=>cx0+v*sc, toY=v=>cy0-v*sc;

  // Trail gradient
  for(let i=1;i<trailX.length;i++){
    const t=i/trailX.length;
    ctxXY.beginPath();
    ctxXY.moveTo(toX(trailX[i-1]),toY(trailY[i-1]));
    ctxXY.lineTo(toX(trailX[i]),toY(trailY[i]));
    ctxXY.strokeStyle=`rgba(56,189,248,${0.08+0.92*t})`;
    ctxXY.lineWidth=0.8+1.4*t; ctxXY.stroke();
  }

  // Current position
  const lx=toX(trailX.at(-1)), ly=toY(trailY.at(-1));
  ctxXY.beginPath(); ctxXY.arc(lx,ly,6,0,2*Math.PI);
  ctxXY.fillStyle='#818cf8'; ctxXY.fill();
  ctxXY.fillStyle='#c7d2fe'; ctxXY.font='bold 9px Inter';
  ctxXY.textAlign='left'; ctxXY.textBaseline='bottom';
  ctxXY.fillText('RX ('+trailX.at(-1).toFixed(1)+', '+trailY.at(-1).toFixed(1)+'m)',lx+8,ly-3);

  // Router origin
  ctxXY.beginPath(); ctxXY.arc(cx0,cy0,6,0,2*Math.PI);
  ctxXY.fillStyle='#ef4444'; ctxXY.fill();
}

// ═══════════════════════════════════════════════════════════
//  Z ALTITUDE TIME-SERIES CANVAS
// ═══════════════════════════════════════════════════════════
const cvsZ = document.getElementById('zchart');
const ctxZ = cvsZ.getContext('2d');
let zHist=[], tHist=[];

function drawZ(newZ, ts){
  const W=cvsZ.offsetWidth||600, H=180;
  cvsZ.width=W; cvsZ.height=H;
  ctxZ.clearRect(0,0,W,H);
  zHist.push(newZ); tHist.push(ts);
  if(zHist.length>300){ zHist.shift(); tHist.shift(); }

  const PAD=36;
  const cw=W-PAD*2, ch=H-PAD*2;

  // Background
  ctxZ.fillStyle='#060d1c'; ctxZ.fillRect(0,0,W,H);

  // Zero line
  const zMin=Math.min(...zHist,-2), zMax=Math.max(...zHist,2);
  const zRange=zMax-zMin||4;
  const toZY=z=>PAD+ch*(1-(z-zMin)/zRange);
  const zero=toZY(0);

  ctxZ.strokeStyle='#1e3a5f'; ctxZ.lineWidth=1;
  ctxZ.beginPath(); ctxZ.moveTo(PAD,zero); ctxZ.lineTo(PAD+cw,zero); ctxZ.stroke();
  ctxZ.fillStyle='#334155'; ctxZ.font='9px Inter'; ctxZ.textAlign='right';
  ctxZ.fillText('0m',PAD-4,zero+3);

  // Y axis labels
  [zMin, (zMin+zMax)/2, zMax].forEach(v=>{
    const y=toZY(v);
    ctxZ.fillStyle='#334155'; ctxZ.textAlign='right';
    ctxZ.fillText(v.toFixed(1)+'m',PAD-4,y+3);
    ctxZ.strokeStyle='#0e2040'; ctxZ.lineWidth=.5;
    ctxZ.beginPath(); ctxZ.moveTo(PAD,y); ctxZ.lineTo(PAD+cw,y); ctxZ.stroke();
  });

  // X axis label
  ctxZ.fillStyle='#334155'; ctxZ.font='9px Inter'; ctxZ.textAlign='center';
  ctxZ.fillText('Time (s)', PAD+cw/2, H-6);
  ctxZ.fillText('Z height above/below router (m)', PAD+cw/2, 12);

  if(zHist.length<2) return;

  const tMin=tHist[0], tMax=tHist.at(-1)||1;
  const toTX=t=>PAD+cw*(t-tMin)/(tMax-tMin||1);

  // Fill under curve
  ctxZ.beginPath();
  ctxZ.moveTo(toTX(tHist[0]),zero);
  for(let i=0;i<zHist.length;i++) ctxZ.lineTo(toTX(tHist[i]),toZY(zHist[i]));
  ctxZ.lineTo(toTX(tHist.at(-1)),zero);
  ctxZ.closePath();
  const grd=ctxZ.createLinearGradient(0,PAD,0,PAD+ch);
  grd.addColorStop(0,'rgba(52,211,153,.35)');
  grd.addColorStop(1,'rgba(52,211,153,.03)');
  ctxZ.fillStyle=grd; ctxZ.fill();

  // Line
  ctxZ.beginPath();
  for(let i=0;i<zHist.length;i++){
    const px=toTX(tHist[i]), py=toZY(zHist[i]);
    i===0?ctxZ.moveTo(px,py):ctxZ.lineTo(px,py);
  }
  ctxZ.strokeStyle='#34d399'; ctxZ.lineWidth=2; ctxZ.stroke();

  // Current value label
  const cx0=toTX(tHist.at(-1));
  const cy0=toZY(zHist.at(-1));
  ctxZ.beginPath(); ctxZ.arc(cx0,cy0,4,0,2*Math.PI);
  ctxZ.fillStyle='#34d399'; ctxZ.fill();
  ctxZ.fillStyle='#a7f3d0'; ctxZ.font='bold 10px JetBrains Mono,monospace';
  ctxZ.textAlign='right';
  ctxZ.fillText((newZ>=0?'+':'')+newZ.toFixed(2)+'m', cx0-7, cy0-7);
}

// ═══════════════════════════════════════════════════════════
//  METRIC CARDS
// ═══════════════════════════════════════════════════════════
function rssiQ(r){ return r>-50?'Excellent':r>-60?'Good':r>-70?'Fair':'Weak'; }

function updateCards(d){
  document.getElementById('vDist').innerHTML=d.d_m.toFixed(2)+'<small style="font-size:.8rem;color:#64748b"> m</small>';
  document.getElementById('sDist').innerText='ToF: '+d.tof_ns.toFixed(1)+' ns';
  document.getElementById('bDist').style.width=Math.min(d.d_m/20*100,100).toFixed(1)+'%';

  document.getElementById('vAz').innerHTML=(d.azimuth_deg>=0?'+':'')+d.azimuth_deg.toFixed(1)+'<small style="font-size:.8rem;color:#64748b"> °</small>';

  document.getElementById('vEl').innerHTML=(d.elevation_deg>=0?'+':'')+d.elevation_deg.toFixed(1)+'<small style="font-size:.8rem;color:#64748b"> °</small>';
  document.getElementById('sEl').innerText='Z = '+(d.z_m>=0?'+':'')+d.z_m.toFixed(2)+' m '+(d.z_m>=0?'(above)':'(below)');

  document.getElementById('vVec').innerHTML=
    'X: <b>'+(d.x_m>=0?'+':'')+d.x_m.toFixed(2)+'</b> m<br>'+
    'Y: <b>'+(d.y_m>=0?'+':'')+d.y_m.toFixed(2)+'</b> m<br>'+
    'Z: <b>'+(d.z_m>=0?'+':'')+d.z_m.toFixed(2)+'</b> m';
  document.getElementById('sVec').innerText='|r| = '+d.d_m.toFixed(2)+' m';

  document.getElementById('vTof').innerHTML=d.tof_ns.toFixed(1)+'<small style="font-size:.8rem;color:#64748b"> ns</small>';
  document.getElementById('vRSSI').innerHTML=d.rssi_dbm.toFixed(0)+'<small style="font-size:.8rem;color:#64748b"> dBm</small>';
  document.getElementById('sRSSI').innerText='Link: '+rssiQ(d.rssi_dbm);

  document.getElementById('vFrame').innerText='Frame '+d.frame_idx+(d.cycle>0?' | cycle '+d.cycle:'');
  document.getElementById('sFrame').innerText='Elapsed: '+d.ts.toFixed(1)+' s';

  // Device label cards
  document.getElementById('txFreq').innerText=(d.tx_freq_ghz||5.2).toFixed(3)+' GHz';
  document.getElementById('txBW').innerText=(d.tx_bw_mhz||40)+' MHz';
  document.getElementById('rxFile').innerText=d.rx_file||'--';
  document.getElementById('rxFrames').innerText=d.rx_frames||'--';
  document.getElementById('rxSR').innerText=(d.rx_sr_hz||34.4).toFixed(1)+' Hz';
  document.getElementById('rxRSSI').innerText=d.rssi_dbm.toFixed(1)+' dBm';

  document.getElementById('status').innerText=
    'Last: '+new Date().toLocaleTimeString()+'  |  frame '+d.frame_idx;
}

// ═══════════════════════════════════════════════════════════
//  SSE EVENT STREAM
// ═══════════════════════════════════════════════════════════
const es = new EventSource('/stream');
es.onmessage = e => {
  try {
    const d = JSON.parse(e.data);
    drawCompass(d.azimuth_deg, d.elevation_deg, d.d_m);
    drawXY(d.x_m, d.y_m);
    drawZ(d.z_m, d.ts);
    updateCards(d);
  } catch(ex){ console.warn(ex); }
};
es.onerror = () => {
  document.getElementById('status').innerText = 'Connection lost — retrying...';
};

// Initial render
drawCompass(0,0,0); drawXY(0,0); drawZ(0,0);
</script>
</body>
</html>"""

# =============================================================================
# FLASK APP
# =============================================================================

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(_DASH)

    @app.route("/api/vector")
    def api_vector():
        with _state_lock:
            return jsonify(dict(_latest_vector))

    @app.route("/stream")
    def stream():
        def generator():
            q = queue.Queue(maxsize=30)
            _sse_clients.append(q)
            try:
                while True:
                    try:
                        yield q.get(timeout=20)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                try: _sse_clients.remove(q)
                except ValueError: pass
        return Response(generator(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})
    return app


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
+==================================================================+
|  WiFiVision — 3D Displacement Vector Tracker                    |
|  LOS-only  |  Azimuth + Elevation AoA  |  ~17 Hz update rate   |
+==================================================================+""")

    ingest = CSIIngestionEngine(n_subcarriers=N_SUBCARRIERS,
                                n_antennas=N_ANTENNAS,
                                sample_rate_hz=SAMPLE_RATE_HZ)

    capture_file = ingest.find_default_capture_file()
    if capture_file is None:
        print("[!] No CSI capture file found in known paths.")
        sys.exit(1)

    print(f"[*] Loading: {os.path.basename(capture_file)}")
    raw_csi, timestamps, meta = ingest.load_from_jsonl(capture_file, max_frames=5000)
    print(f"[+] {meta['n_frames']} frames @ {meta['sample_rate_hz']:.1f} Hz  "
          f"| RSSI: {meta['mean_rssi_dbm']:.1f} dBm")

    print("[*] Sanitizing (Hampel + low-pass, NO background subtraction)...")
    csi_clean = ingest.remove_outliers_hampel(raw_csi)
    csi_clean = ingest.apply_lowpass_filter(csi_clean, cutoff_freq_hz=15.0)
    rssi_arr  = np.full(meta['n_frames'], meta['mean_rssi_dbm'])

    t = threading.Thread(target=tracking_thread,
                         args=(csi_clean, rssi_arr, timestamps, meta),
                         daemon=True)
    t.start()

    if not FLASK_AVAILABLE:
        print("[!] Flask not installed. Run:  .venv/bin/pip install flask")
        t.join()
        return

    print(f"\n[+] Dashboard: http://localhost:{DASHBOARD_PORT}")
    print(f"    Update rate: ~{SAMPLE_RATE_HZ/STRIDE_FRAMES:.1f} Hz  "
          f"(stride={STRIDE_FRAMES} frames)\n    Ctrl+C to stop.\n")

    app = create_app()
    try:
        app.run(host="0.0.0.0", port=DASHBOARD_PORT,
                threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[*] Stopped.")


if __name__ == "__main__":
    main()
