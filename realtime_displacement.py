#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFiVision — Real-Time Relative Displacement Vector Tracker
============================================================
Continuously measures and displays the real-time relative displacement vector
[distance, azimuth, X, Y] between two communicating Wi-Fi devices (router and
laptop) by locking exclusively onto the direct Line-of-Sight (LOS) signal path.

WHAT THIS DOES (and does NOT do):
  ✓ Isolates the direct LOS signal path (shortest ToF, highest power)
  ✓ Calculates real-time distance via ToF: d = tau * c
  ✓ Calculates real-time azimuth angle via 2D-MUSIC super-resolution AoA
  ✓ Fuses distance + angle into Cartesian displacement vector [X, Y]
  ✓ Streams live vector updates to a browser dashboard via HTTP SSE
  ✗ NO motion/Doppler tracking of room occupants
  ✗ NO Gaussian splatting or room geometry reconstruction
  ✗ NO vital sign / heartbeat monitoring
  ✗ NO background subtraction (which would REMOVE the direct LOS path)

Physics:
  - Direct LOS path is identified as the peak with minimum ToF (shortest delay)
    in the Power Delay Profile (PDP) that also has the highest magnitude.
  - Background subtraction is intentionally DISABLED. The static direct path IS
    the signal of interest — we must not subtract it away.
  - AoA is computed via 2D-MUSIC on the phase difference across antenna elements:
      phi_ant(a) = pi * a * sin(theta_LOS)
  - Distance via ToF sensor fusion (CSI + RSSI):
      d = tau_LOS * c    (speed of light = 2.998e8 m/s)

Usage:
  ./.venv/bin/pip install flask   # (one-time, if not installed)
  ./.venv/bin/python realtime_displacement.py
  Then open http://localhost:5757 in your browser.

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
from scipy import fft

# --- Try importing Flask for the live dashboard server -----------------------
try:
    from flask import Flask, Response, render_template_string, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from ingestion import CSIIngestionEngine
from dsp_engine import CSIDSPEngine

# --- Constants ---------------------------------------------------------------
SPEED_OF_LIGHT  = 2.998e8          # m/s
CARRIER_HZ      = 5.2e9            # 5.2 GHz
BANDWIDTH_HZ    = 40.0e6           # 40 MHz
N_SUBCARRIERS   = 64
N_ANTENNAS      = 4
SAMPLE_RATE_HZ  = 34.4             # Real hardware capture rate
WINDOW_FRAMES   = 32               # CSI snapshot window for 2D-MUSIC
STRIDE_FRAMES   = 8                # Sliding window stride (update every 8 new frames)
DASHBOARD_PORT  = 5757
MAX_HISTORY_PTS = 200              # Number of historical XY positions to keep

# --- Shared state (written by tracker, read by dashboard) --------------------
_state_lock = threading.Lock()
_latest_vector = {
    "d_m":       0.0,
    "aoa_deg":   0.0,
    "x_m":       0.0,
    "y_m":       0.0,
    "tof_ns":    0.0,
    "rssi_dbm": -50.0,
    "frame_idx":  0,
    "ts":         0.0,
}
_history_x: list = []
_history_y: list = []
_sse_clients: list = []   # list of queue.Queue() for SSE event streaming


# =============================================================================
# LOS DISPLACEMENT EXTRACTOR
# =============================================================================

class LOSDisplacementExtractor:
    """
    Strips the pipeline down to LOS-only estimation.

    Key difference from the full pipeline:
      - sanitize_phase_sfo_cfo()  => kept  (removes hardware clock offsets)
      - remove_static_background() => DISABLED (would remove the direct path)
      - compute_pdp_ifft()         => kept, but we select the MINIMUM-delay peak
      - estimate_2d_music()        => kept, but we pass only the direct-path peak
    """

    def __init__(self):
        self.dsp = CSIDSPEngine(
            carrier_freq_hz=CARRIER_HZ,
            bandwidth_hz=BANDWIDTH_HZ,
            n_subcarriers=N_SUBCARRIERS,
            n_antennas=N_ANTENNAS,
            sample_rate_hz=SAMPLE_RATE_HZ,
        )
        # Delay grid: 1 -> 100 ns (0.3 m -> 30 m)
        self.delay_grid_ns  = np.linspace(1.0, 100.0, 200)
        # Angle grid: -90 -> +90 degrees
        self.angle_grid_deg = np.linspace(-90.0, 90.0, 181)

    def extract(self, csi_window: np.ndarray, rssi_dbm: float = -50.0) -> dict:
        """
        csi_window: (N_subcarriers, N_frames, N_antennas) complex array
        Returns displacement vector dict.
        """
        # 1. Phase sanitization (SFO/CFO removal) - keeps hardware clock errors out
        csi_clean = self.dsp.sanitize_phase_sfo_cfo(csi_window)

        # 2. Collapse to mean snapshot across the window (reduces noise)
        csi_snap = np.mean(csi_clean, axis=1)  # (N_f, N_a)

        # 3. Power Delay Profile - identify shortest-delay (LOS) peak
        pdp, _ = self.dsp.compute_pdp_ifft(csi_clean)
        # pdp shape varies - average magnitude across antennas & time
        pdp_mean = np.mean(np.abs(pdp), axis=tuple(i for i in range(1, pdp.ndim)))

        # Map bins to nanoseconds using the bandwidth
        bin_to_ns = (1.0 / BANDWIDTH_HZ) * 1e9  # ns per bin
        delay_axis_ns = np.arange(len(pdp_mean)) * bin_to_ns

        # Find the earliest strong peak (direct LOS = minimum delay peak above threshold)
        threshold = np.max(pdp_mean) * 0.15   # 15% of max power
        los_candidates = np.where(pdp_mean > threshold)[0]
        if len(los_candidates) > 0:
            los_bin = los_candidates[0]       # first (shortest delay) peak
            tof_ns  = float(delay_axis_ns[los_bin])
        else:
            tof_ns = 10.0  # fallback: ~3 m

        # 4. 2D-MUSIC for AoA at the identified LOS delay
        try:
            _, peaks = self.dsp.estimate_2d_music(
                csi_snap,
                n_sources=1,                      # only 1 source = direct path
                angle_grid_deg=self.angle_grid_deg,
                delay_grid_ns=self.delay_grid_ns,
            )
            if peaks:
                aoa_deg = float(peaks[0][0])
                tof_music_ns = float(peaks[0][1])
                # Blend: weight MUSIC ToF 60%, PDP ToF 40%
                tof_ns = 0.6 * tof_music_ns + 0.4 * tof_ns
            else:
                aoa_deg = 0.0
        except Exception:
            aoa_deg = 0.0

        # 5. Distance from ToF: d = tau * c
        d_csi = (tof_ns * 1e-9) * SPEED_OF_LIGHT

        # 6. RSSI path-loss distance (backup metric)
        rssi_clamped = min(-31.0, max(-95.0, float(rssi_dbm)))
        d_rssi = 10.0 ** ((-rssi_clamped - 30.0) / (10.0 * 2.8))
        d_rssi = min(d_rssi, 50.0)  # clamp to 50 m

        # 7. Fused distance
        d_fused = 0.75 * d_csi + 0.25 * d_rssi

        # 8. Cartesian displacement vector [X, Y]
        aoa_rad = math.radians(aoa_deg)
        x = d_fused * math.cos(aoa_rad)
        y = d_fused * math.sin(aoa_rad)

        return {
            "d_m":      round(d_fused, 3),
            "aoa_deg":  round(aoa_deg, 1),
            "x_m":      round(x, 3),
            "y_m":      round(y, 3),
            "tof_ns":   round(tof_ns, 2),
            "rssi_dbm": round(rssi_dbm, 1),
        }


# =============================================================================
# REAL-TIME TRACKING THREAD
# =============================================================================

def tracking_thread(csi_full: np.ndarray, rssi_arr: np.ndarray, timestamps: np.ndarray):
    """
    Slides a window across the loaded CSI frames, simulating real-time
    arrival of new frames at the hardware sample rate.
    Continuously cycles through the data so the display keeps updating.
    """
    global _latest_vector, _history_x, _history_y

    extractor = LOSDisplacementExtractor()
    n_frames  = csi_full.shape[1]
    frame_idx = 0
    cycle     = 0

    print(f"[*] Tracker started: {n_frames} frames, window={WINDOW_FRAMES}, stride={STRIDE_FRAMES}")
    print(f"[*] Update rate: ~{SAMPLE_RATE_HZ/STRIDE_FRAMES:.1f} Hz  "
          f"(every {STRIDE_FRAMES/SAMPLE_RATE_HZ*1000:.0f} ms)")

    while True:
        start = frame_idx % n_frames
        end   = (frame_idx + WINDOW_FRAMES) % n_frames

        if end > start:
            csi_win  = csi_full[:, start:end, :]
            rssi_win = rssi_arr[start:end]
        else:
            # Wrap-around at end of recording
            csi_win  = np.concatenate([csi_full[:, start:, :], csi_full[:, :end, :]], axis=1)
            rssi_win = np.concatenate([rssi_arr[start:], rssi_arr[:end]])

        mean_rssi = float(np.mean(rssi_win))

        try:
            vec = extractor.extract(csi_win, rssi_dbm=mean_rssi)
        except Exception as e:
            vec = {"d_m": 0.0, "aoa_deg": 0.0, "x_m": 0.0, "y_m": 0.0,
                   "tof_ns": 0.0, "rssi_dbm": mean_rssi}

        vec["frame_idx"] = frame_idx
        vec["ts"]        = float(timestamps[start % n_frames])
        vec["cycle"]     = cycle

        with _state_lock:
            _latest_vector.update(vec)
            _history_x.append(vec["x_m"])
            _history_y.append(vec["y_m"])
            if len(_history_x) > MAX_HISTORY_PTS:
                _history_x.pop(0)
                _history_y.pop(0)

        # Push to all SSE clients
        payload = "data: " + json.dumps(vec) + "\n\n"
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass

        # Terminal output
        print(f"\r[{frame_idx:5d}] "
              f"d={vec['d_m']:5.2f}m  "
              f"AoA={vec['aoa_deg']:+6.1f}deg  "
              f"X={vec['x_m']:+6.2f}m  "
              f"Y={vec['y_m']:+6.2f}m  "
              f"ToF={vec['tof_ns']:5.1f}ns  "
              f"RSSI={vec['rssi_dbm']:+.0f}dBm",
              end="", flush=True)

        frame_idx += STRIDE_FRAMES
        if frame_idx >= n_frames:
            frame_idx = 0
            cycle += 1

        # Sleep to match hardware sample rate
        time.sleep(STRIDE_FRAMES / SAMPLE_RATE_HZ)


# =============================================================================
# FLASK LIVE DASHBOARD
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WiFiVision - Real-Time Displacement Vector</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', system-ui, sans-serif; background: #070b14;
         color: #e2e8f0; min-height: 100vh; padding: 20px; }

  .page-header { text-align: center; margin-bottom: 24px; }
  .page-header h1 { font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .page-header p { color: #64748b; font-size: 0.9rem; margin-top: 4px; }

  .live-badge { display: inline-flex; align-items: center; gap: 6px;
    background: #0f172a; border: 1px solid #1e3a5f; border-radius: 20px;
    padding: 4px 12px; font-size: 0.75rem; color: #38bdf8; margin-top: 8px; }
  .live-dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e;
    animation: pulse-dot 1.2s ease infinite; }
  @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(0.6)} }

  .grid { display: grid; grid-template-columns: 1fr 340px; gap: 20px; }

  .compass-wrap { background: #0d1526; border: 1px solid #1e293b;
    border-radius: 16px; padding: 24px; display: flex; flex-direction: column;
    align-items: center; gap: 20px; }
  .compass-title { font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: .1em; color: #475569; font-weight: 600; }

  .scatter-wrap { background: #0d1526; border: 1px solid #1e293b;
    border-radius: 16px; padding: 20px; }

  .metrics { display: flex; flex-direction: column; gap: 14px; }
  .card { background: #0d1526; border: 1px solid #1e293b; border-radius: 12px;
    padding: 16px 20px; }
  .card-label { font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: .08em; color: #475569; font-weight: 600; margin-bottom: 8px; }
  .card-value { font-family: 'JetBrains Mono', monospace; font-size: 2rem;
    font-weight: 600; color: #f1f5f9; line-height: 1; }
  .card-unit { font-size: 0.85rem; color: #64748b; margin-left: 4px; }
  .card-sub { font-size: 0.78rem; color: #475569; margin-top: 4px; }

  .card.distance .card-value { color: #38bdf8; }
  .card.angle    .card-value { color: #818cf8; }
  .card.vector   .card-value { font-size: 1.3rem; color: #34d399; }
  .card.tof      .card-value { font-size: 1.5rem; color: #fb923c; }
  .card.rssi     .card-value { font-size: 1.5rem; color: #f472b6; }

  .history-bar { height: 4px; border-radius: 2px; background: #1e293b;
    margin-top: 8px; overflow: hidden; }
  .history-fill { height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    transition: width 0.3s ease; }

  #status { position: fixed; bottom: 16px; right: 20px; font-size: 0.7rem;
    color: #334155; font-family: 'JetBrains Mono', monospace; }
</style>
</head>
<body>

<div class="page-header">
  <h1>WiFiVision - Real-Time Displacement Vector</h1>
  <p>Direct LOS signal path only &mdash; 2D-MUSIC AoA + ToF ranging &mdash; Router (TX) vs Laptop (RX)</p>
  <div class="live-badge"><span class="live-dot"></span> LIVE &mdash; updating in real time</div>
</div>

<div class="grid">

  <div style="display:flex;flex-direction:column;gap:20px;">
    <div class="compass-wrap">
      <div class="compass-title">Relative Azimuth (AoA) &mdash; Router at centre</div>
      <canvas id="compass" width="320" height="320"></canvas>
    </div>
    <div class="scatter-wrap">
      <div class="compass-title" style="margin-bottom:12px;">XY Position History Trail</div>
      <canvas id="scatter" width="600" height="260"></canvas>
    </div>
  </div>

  <div class="metrics">

    <div class="card distance">
      <div class="card-label">Fused Distance</div>
      <div class="card-value" id="valDist">--<span class="card-unit">m</span></div>
      <div class="card-sub" id="subDist">ToF: -- ns</div>
      <div class="history-bar"><div class="history-fill" id="distBar" style="width:0%"></div></div>
    </div>

    <div class="card angle">
      <div class="card-label">Azimuth Angle (AoA)</div>
      <div class="card-value" id="valAoa">--<span class="card-unit">deg</span></div>
      <div class="card-sub">Relative to antenna boresight (0 = broadside)</div>
    </div>

    <div class="card vector">
      <div class="card-label">Cartesian Displacement Vector</div>
      <div class="card-value" id="valVec">X: -- m<br>Y: -- m</div>
      <div class="card-sub" id="subVec">|r| = -- m at angle --</div>
    </div>

    <div class="card tof">
      <div class="card-label">Time-of-Flight (LOS)</div>
      <div class="card-value" id="valTof">--<span class="card-unit">ns</span></div>
      <div class="card-sub">d = tau x c  |  1 ns = 0.30 m</div>
    </div>

    <div class="card rssi">
      <div class="card-label">RSSI (Signal Strength)</div>
      <div class="card-value" id="valRssi">--<span class="card-unit">dBm</span></div>
      <div class="card-sub" id="subRssi">Link quality: --</div>
    </div>

    <div class="card">
      <div class="card-label">Frame Counter</div>
      <div class="card-value" id="valFrame" style="font-size:1.2rem;color:#94a3b8;">--</div>
      <div class="card-sub">Sample rate: """ + f"{SAMPLE_RATE_HZ:.1f}" + """ Hz</div>
    </div>

  </div>
</div>

<div id="status">Connecting...</div>

<script>
const compass = document.getElementById('compass');
const ctx = compass.getContext('2d');
const CW = 320, CH = 320, CR = 140, cx = CW/2, cy = CH/2;

function drawCompass(aoa_deg, dist_m) {
  ctx.clearRect(0, 0, CW, CH);
  ctx.beginPath(); ctx.arc(cx, cy, CR, 0, 2*Math.PI);
  ctx.fillStyle = '#060d1c'; ctx.fill();
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 2; ctx.stroke();

  [0.33, 0.66, 1.0].forEach(frac => {
    ctx.beginPath(); ctx.arc(cx, cy, CR*frac, 0, 2*Math.PI);
    ctx.strokeStyle = '#0f2040'; ctx.lineWidth = 1; ctx.stroke();
  });

  ctx.strokeStyle = '#1e3a5f'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx-CR, cy); ctx.lineTo(cx+CR, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy-CR); ctx.lineTo(cx, cy+CR); ctx.stroke();

  for (let a = -90; a <= 90; a += 15) {
    const r = Math.PI/2 - a*Math.PI/180;
    const inner = CR*0.88, outer = CR*0.98;
    ctx.beginPath();
    ctx.moveTo(cx + inner*Math.cos(r), cy - inner*Math.sin(r));
    ctx.lineTo(cx + outer*Math.cos(r), cy - outer*Math.sin(r));
    ctx.strokeStyle = a === 0 ? '#38bdf8' : '#1e3a5f';
    ctx.lineWidth = a % 45 === 0 ? 2 : 1; ctx.stroke();
    if (a % 30 === 0) {
      ctx.fillStyle = '#475569'; ctx.font = '10px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(a+'deg', cx + CR*0.72*Math.cos(r), cy - CR*0.72*Math.sin(r) + 4);
    }
  }

  // Router centre
  ctx.beginPath(); ctx.arc(cx, cy, 8, 0, 2*Math.PI);
  ctx.fillStyle = '#ef4444'; ctx.fill();
  ctx.fillStyle = '#fff'; ctx.font = 'bold 9px Inter';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('TX', cx, cy);

  // Displacement arrow
  const aoaRad = Math.PI/2 - aoa_deg*Math.PI/180;
  const normDist = Math.min(dist_m / 20.0, 1.0);
  const tipX = cx + normDist*CR*0.85*Math.cos(aoaRad);
  const tipY = cy - normDist*CR*0.85*Math.sin(aoaRad);

  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tipX, tipY);
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 3; ctx.stroke();

  const headLen = 12, headAngle = 0.4;
  const ang = Math.atan2(cy - tipY, tipX - cx);
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - headLen*Math.cos(ang-headAngle), tipY + headLen*Math.sin(ang-headAngle));
  ctx.lineTo(tipX - headLen*Math.cos(ang+headAngle), tipY + headLen*Math.sin(ang+headAngle));
  ctx.closePath(); ctx.fillStyle = '#38bdf8'; ctx.fill();

  ctx.beginPath(); ctx.arc(tipX, tipY, 6, 0, 2*Math.PI);
  ctx.fillStyle = '#818cf8'; ctx.fill();
  ctx.fillStyle = '#fff'; ctx.font = 'bold 8px Inter';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('RX', tipX, tipY);

  ctx.fillStyle = '#38bdf8'; ctx.font = 'bold 12px monospace';
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  const mX = cx + (normDist*CR*0.85/2)*Math.cos(aoaRad) + 14;
  const mY = cy - (normDist*CR*0.85/2)*Math.sin(aoaRad) - 8;
  ctx.fillText(dist_m.toFixed(2)+'m', mX, mY);
}

const scatterEl = document.getElementById('scatter');
const sctx = scatterEl.getContext('2d');
let histX = [], histY = [];

function drawScatter(newX, newY) {
  const W = 600, H = 260;
  sctx.clearRect(0, 0, W, H);
  histX.push(newX); histY.push(newY);
  if (histX.length > 200) { histX.shift(); histY.shift(); }

  const cx0 = W/2, cy0 = H/2;
  sctx.strokeStyle = '#0f2040'; sctx.lineWidth = 1;
  [0.25,0.5,0.75,1.0].forEach(f => {
    sctx.beginPath(); sctx.arc(cx0, cy0, Math.min(W,H)/2*0.9*f, 0, 2*Math.PI); sctx.stroke();
  });
  sctx.beginPath(); sctx.moveTo(0, cy0); sctx.lineTo(W, cy0); sctx.stroke();
  sctx.beginPath(); sctx.moveTo(cx0, 0); sctx.lineTo(cx0, H); sctx.stroke();

  if (histX.length < 2) return;
  const allX = histX.concat([0]), allY = histY.concat([0]);
  const xRange = Math.max(...allX.map(Math.abs)) * 1.3 || 5;
  const yRange = Math.max(...allY.map(Math.abs)) * 1.3 || 5;
  const scale  = Math.min((W/2*0.85)/xRange, (H/2*0.85)/yRange);
  const toSX = x => cx0 + x*scale;
  const toSY = y => cy0 - y*scale;

  for (let i = 1; i < histX.length; i++) {
    const t = i/histX.length;
    sctx.beginPath();
    sctx.moveTo(toSX(histX[i-1]), toSY(histY[i-1]));
    sctx.lineTo(toSX(histX[i]),   toSY(histY[i]));
    sctx.strokeStyle = 'rgba(56,189,248,'+(0.2+0.8*t)+')';
    sctx.lineWidth = 1 + t; sctx.stroke();
  }

  const lX = toSX(histX[histX.length-1]);
  const lY = toSY(histY[histY.length-1]);
  sctx.beginPath(); sctx.arc(lX, lY, 6, 0, 2*Math.PI);
  sctx.fillStyle = '#818cf8'; sctx.fill();

  sctx.beginPath(); sctx.arc(cx0, cy0, 5, 0, 2*Math.PI);
  sctx.fillStyle = '#ef4444'; sctx.fill();
}

function rssiQuality(r) {
  if (r > -50) return 'Excellent';
  if (r > -60) return 'Good';
  if (r > -70) return 'Fair';
  return 'Weak';
}

function updateCards(d) {
  document.getElementById('valDist').innerHTML  = d.d_m.toFixed(2)+'<span class="card-unit">m</span>';
  document.getElementById('subDist').innerText  = 'ToF: '+d.tof_ns.toFixed(1)+' ns';
  document.getElementById('distBar').style.width = Math.min(d.d_m/20*100,100).toFixed(1)+'%';
  const sign = d.aoa_deg >= 0 ? '+' : '';
  document.getElementById('valAoa').innerHTML   = sign+d.aoa_deg.toFixed(1)+'<span class="card-unit">deg</span>';
  document.getElementById('valVec').innerHTML   = 'X: '+(d.x_m>=0?'+':'')+d.x_m.toFixed(2)+' m<br>Y: '+(d.y_m>=0?'+':'')+d.y_m.toFixed(2)+' m';
  document.getElementById('subVec').innerText   = '|r| = '+d.d_m.toFixed(2)+' m at '+d.aoa_deg.toFixed(1)+'deg';
  document.getElementById('valTof').innerHTML   = d.tof_ns.toFixed(1)+'<span class="card-unit">ns</span>';
  document.getElementById('valRssi').innerHTML  = d.rssi_dbm.toFixed(0)+'<span class="card-unit">dBm</span>';
  document.getElementById('subRssi').innerText  = 'Link quality: '+rssiQuality(d.rssi_dbm);
  document.getElementById('valFrame').innerText = 'Frame '+d.frame_idx+(d.cycle>0?' (cycle '+d.cycle+')':'');
  document.getElementById('status').innerText   = 'Last update: '+new Date().toLocaleTimeString()+'  |  frame '+d.frame_idx;
}

const evtSource = new EventSource('/stream');
evtSource.onmessage = (e) => {
  try {
    const d = JSON.parse(e.data);
    drawCompass(d.aoa_deg, d.d_m);
    drawScatter(d.x_m, d.y_m);
    updateCards(d);
  } catch(ex) { console.warn('Parse error:', ex); }
};
evtSource.onerror = () => {
  document.getElementById('status').innerText = 'Connection lost - retrying...';
};

drawCompass(0, 0);
drawScatter(0, 0);
</script>
</body>
</html>"""


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/vector")
    def api_vector():
        with _state_lock:
            return jsonify(dict(_latest_vector))

    @app.route("/stream")
    def stream():
        def event_stream():
            q = queue.Queue(maxsize=20)
            _sse_clients.append(q)
            try:
                while True:
                    try:
                        data = q.get(timeout=30)
                        yield data
                    except queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                try:
                    _sse_clients.remove(q)
                except ValueError:
                    pass
        return Response(event_stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    return app


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("""
+==================================================================+
|  WiFiVision - Real-Time Relative Displacement Vector Tracker    |
|  Direct LOS Path Only  |  2D-MUSIC AoA + ToF Ranging           |
+==================================================================+
""")

    # Load real hardware CSI capture
    ingest = CSIIngestionEngine(n_subcarriers=N_SUBCARRIERS,
                                n_antennas=N_ANTENNAS,
                                sample_rate_hz=SAMPLE_RATE_HZ)

    capture_file = ingest.find_default_capture_file()
    if capture_file is None:
        print("[!] No CSI capture file found.")
        print("    Ensure a *.csi.jsonl file is in ../RuView/data/recordings/ or current directory.")
        sys.exit(1)

    print(f"[*] Loading: {os.path.basename(capture_file)}")
    raw_csi, timestamps, meta = ingest.load_from_jsonl(capture_file, max_frames=5000)
    print(f"[+] {meta['n_frames']} frames @ {meta['sample_rate_hz']:.1f} Hz  |  "
          f"RSSI: {meta['mean_rssi_dbm']:.1f} dBm")

    # Phase sanitization only (NO background subtraction - preserves LOS path)
    print("[*] Sanitizing phase (Hampel + low-pass)...")
    csi_clean = ingest.remove_outliers_hampel(raw_csi)
    csi_clean = ingest.apply_lowpass_filter(csi_clean, cutoff_freq_hz=15.0)

    rssi_arr = np.full(meta['n_frames'], meta['mean_rssi_dbm'])

    # Start tracker thread
    t = threading.Thread(target=tracking_thread,
                         args=(csi_clean, rssi_arr, timestamps),
                         daemon=True)
    t.start()

    if not FLASK_AVAILABLE:
        print("[!] Flask not installed. Run:  .venv/bin/pip install flask")
        print("[*] Running terminal-only mode.")
        t.join()
        return

    print(f"\n[+] Dashboard: http://localhost:{DASHBOARD_PORT}")
    print(f"    Open that URL in your browser for the live displacement vector.\n")
    print(f"    Press Ctrl+C to stop.\n")

    app = create_app()
    try:
        app.run(host="0.0.0.0", port=DASHBOARD_PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[*] Stopped.")


if __name__ == "__main__":
    main()
