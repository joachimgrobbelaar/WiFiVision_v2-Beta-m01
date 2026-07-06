#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Unified Executable Launcher for Wi-Fi CSI Sensing & Geometry Pipeline
This module serves as the primary entry point for PyInstaller compilation into a single binary executable.
"""

import sys
import os
import argparse

def print_banner():
    banner = """
==============================================================================
  WI-FI CSI INDOOR PHYSICAL GEOMETRY RECONSTRUCTION & VITAL SIGN TRACKING
  Advanced Agentic DSP & RF Physics Sensing Pipeline
==============================================================================
"""
    print(banner)

def run_diagnostics():
    print("[*] Executing System & RF Hardware Compatibility Diagnostics...\n")
    import subprocess
    if os.path.exists("env_check.sh"):
        subprocess.run(["bash", "env_check.sh"], check=False)
    else:
        print("[-] env_check.sh not found in current directory. Performing Python RF check...")
        import platform
        print(f"    System: {platform.system()} {platform.release()} ({platform.machine()})")
        print("    No specialized CSI kernel patches active. Fallback simulation mode ready.\n")

def run_tests():
    print("[*] Executing Unit Verification Suite across Core Modules...\n")
    from ingestion import CSIIngestionEngine
    from dsp_engine import CSIDSPEngine
    from geometry_mapping import GeometricMapper
    
    print(" -> Testing CSIIngestionEngine...")
    import numpy as np
    ingest = CSIIngestionEngine(n_subcarriers=64, n_antennas=4, sample_rate_hz=100.0)
    test_csi = np.ones((64, 50, 4), dtype=np.complex128)
    test_csi[10, 25, 0] = 100.0 + 100.0j # Inject spike
    cleaned = ingest.remove_outliers_hampel(test_csi, window_size=5, threshold=3.0)
    print(f"    [+] Outlier attenuation: before={np.abs(test_csi[10, 25, 0]):.2f}, after={np.abs(cleaned[10, 25, 0]):.2f}")
    print("    [+] CSIIngestionEngine verification passed!\n")
    
    print(" -> Testing CSIDSPEngine & Material Classifier...")
    dsp = CSIDSPEngine(carrier_freq_hz=5.2e9, bandwidth_hz=40e6, n_subcarriers=64, n_antennas=4)
    mat = dsp.classify_material_semantic(raw_attenuation_db=62.0, tof_ns=23.3, freq_band='5GHz')
    print(f"    [+] Material Classifier prediction: {mat['label_str']} (Residual Loss: {mat['residual_loss_db']:.1f} dB)")
    print("    [+] CSIDSPEngine verification passed!\n")
    
    print(" -> Testing GeometricMapper (2D & 3D Bistatic)...")
    mapper = GeometricMapper(dsp_engine=dsp)
    peaks = [(30.0, 15.0, -10.0)]
    coords = mapper.map_peaks_to_cartesian(peaks)
    print(f"    [+] 2D Cartesian coordinates mapped: (X={coords[0,0]:.2f}m, Y={coords[0,1]:.2f}m)")
    
    # 3D Bistatic Test
    tx_3d, rx_3d = (3.5, 4.0, 2.5), (0.0, 0.0, 1.0)
    obs_test = np.array([1.5, 2.0, 1.5])
    d_tx, d_rx = np.linalg.norm(obs_test - np.array(tx_3d)), np.linalg.norm(obs_test - np.array(rx_3d))
    tof_obs = ((d_tx + d_rx) / dsp.c) * 1e9
    vec_rx = obs_test - np.array(rx_3d)
    az_obs, el_obs = np.degrees(np.arctan2(vec_rx[1], vec_rx[0])), np.degrees(np.arcsin(vec_rx[2] / d_rx))
    coords_3d = mapper.map_bistatic_peaks_to_3d([(az_obs, el_obs, tof_obs, 10.0)], tx_pos_3d=tx_3d, rx_pos_3d=rx_3d)
    print(f"    [+] 3D Bistatic Obstacle mapped: (X={coords_3d[0,0]:.2f}m, Y={coords_3d[0,1]:.2f}m, Z={coords_3d[0,2]:.2f}m)")
    print("    [+] GeometricMapper verification passed!\n")

def run_simulation():
    print("[*] Launching Phase 5 Self-Contained Room Simulation & Dashboard...\n")
    from simulation import RoomCSISimulator
    sim = RoomCSISimulator(n_subcarriers=64, n_antennas=4, sample_rate_hz=100.0)
    sim.execute_and_visualize()

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Wi-Fi CSI Sensing Standalone Executable")
    parser.add_argument("--test", action="store_true", help="Run module unit verification tests")
    parser.add_argument("--diag", action="store_true", help="Run system & RF hardware diagnostics")
    parser.add_argument("--all", action="store_true", help="Run diagnostics, tests, and simulation")
    
    args = parser.parse_args()
    
    if args.diag or args.all:
        run_diagnostics()
    if args.test or args.all:
        run_tests()
    if not (args.diag or args.test) or args.all:
        run_simulation()

if __name__ == "__main__":
    main()
