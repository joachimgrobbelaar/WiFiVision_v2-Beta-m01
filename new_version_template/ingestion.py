#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Automated Data Extraction & Ingestion Pipeline
Wi-Fi CSI Indoor Physical Geometry Reconstruction Pipeline

This module implements the ingestion and pre-processing engine for raw binary
Channel State Information (CSI) packets extracted from Linux kernel wireless
character devices, netlink sockets, or synthetic PCAP/numpy streams.

Physics & Math Principles:
1. Complex Baseband Representation: Raw CSI packets encode orthogonal frequency
   division multiplexing (OFDM) subcarriers as in-phase (I) and quadrature (Q)
   signed integers. The baseband channel frequency response is constructed as:
       H(f, t, a) = I(f, t, a) + j * Q(f, t, a)
   where f is the subcarrier index, t is the time frame index, and a is the
   receive antenna index.
2. Hampel Outlier Rejection: Automatic Gain Control (AGC) transients and bursty
   RF interference introduce non-Gaussian impulsive noise. We remove these
   outliers across time t using Median Absolute Deviation (MAD):
       MAD = median(|x_t - median(x)|)
   Samples exceeding threshold * MAD are replaced by the windowed median.
3. Low-Pass Filtering: Thermal receiver noise adds high-frequency fluctuations.
   We apply a zero-phase Butterworth low-pass filter across time packets t to
   suppress noise while preserving structural reflection Doppler variations.
"""

import numpy as np
from scipy import signal
import struct
import os
import json
import glob
from typing import Tuple, Optional, Union, Dict, Any, List

class CSIIngestionEngine:
    """
    Ingestion and preprocessing engine for multi-dimensional Wi-Fi CSI data.
    Manages raw binary unpacking, outlier suppression, low-pass filtering,
    and missing packet interpolation.
    """
    def __init__(self, n_subcarriers: int = 64, n_antennas: int = 4, sample_rate_hz: float = 100.0):
        """
        Initialize the ingestion engine with expected dimensions.
        
        Args:
            n_subcarriers: Number of OFDM subcarriers per packet (e.g., 30 for Intel 5300, 64/128 for Nexmon).
            n_antennas: Number of receive antennas in the array.
            sample_rate_hz: Expected packet sampling frequency in Hz.
        """
        self.n_subcarriers = n_subcarriers
        self.n_antennas = n_antennas
        self.sample_rate_hz = sample_rate_hz

    def unpack_raw_binary_stream(self, binary_data: bytes, n_packets: int) -> np.ndarray:
        """
        Unpack raw binary bytes (e.g., from character device /dev/csi or netlink socket)
        into a complex multi-dimensional NumPy array H(f, t, a).
        
        Args:
            binary_data: Byte buffer containing packed 16-bit signed integer I/Q pairs.
            n_packets: Expected number of time packets encoded in the buffer.
            
        Returns:
            np.ndarray of complex128 with shape (n_subcarriers, n_packets, n_antennas).
        """
        expected_samples = self.n_subcarriers * n_packets * self.n_antennas
        expected_bytes = expected_samples * 4 # 2 bytes I + 2 bytes Q (int16)
        
        if len(binary_data) < expected_bytes:
            raise ValueError(f"Insufficient binary data: expected {expected_bytes} bytes, got {len(binary_data)}")
        
        # Unpack int16 integers as alternating I and Q samples
        raw_ints = np.frombuffer(binary_data[:expected_bytes], dtype=np.int16)
        i_samples = raw_ints[0::2].astype(np.float64)
        q_samples = raw_ints[1::2].astype(np.float64)
        
        # Construct complex channel matrix H
        complex_samples = i_samples + 1j * q_samples
        
        # Reshape to (n_packets, n_subcarriers, n_antennas) then transpose to (f, t, a)
        h_matrix = complex_samples.reshape((n_packets, self.n_subcarriers, self.n_antennas))
        h_matrix = np.transpose(h_matrix, (1, 0, 2))
        return h_matrix

    def find_default_capture_file(self) -> Optional[str]:
        """
        Automatically discover real Wi-Fi CSI signal capture files across standard recording directories and workspaces.
        """
        search_paths = [
            "/home/m/WebApps/WifiVision/RuView/data/recordings/pretrain-1775182186.csi.jsonl",
            "/home/m/WebApps/WifiVision/RuView/data/recordings/overnight-1775217646.csi.jsonl",
            "../RuView/data/recordings/pretrain-1775182186.csi.jsonl",
            "../RuView/data/recordings/overnight-1775217646.csi.jsonl",
            "./data/recordings/pretrain-1775182186.csi.jsonl",
            "./data/recordings/overnight-1775217646.csi.jsonl",
        ]
        for path in search_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        for ext in ["*.jsonl", "*.dat", "*.csv"]:
            matches = glob.glob(ext) + glob.glob(f"../**/{ext}", recursive=True)
            if matches:
                return os.path.abspath(matches[0])
        return None

    def load_from_jsonl(self, file_path: str, max_frames: int = 3000, target_node: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load and decode real hardware Wi-Fi CSI signal data from a JSONL/JSON capture stream.
        Extracts alternating I/Q samples from hex payloads, timestamps, RSSI, and hardware vital telemetry.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Real CSI capture file not found: {file_path}")

        frames_csi = []
        timestamps = []
        rssi_vals = []
        hw_vitals = {"breathing_bpm": [], "heartrate_bpm": [], "motion_energy": [], "timestamp": []}
        node_counts = {}

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_str in f:
                line_str = line_str.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                except Exception:
                    continue

                node_id = record.get("node_id", 1)
                node_counts[node_id] = node_counts.get(node_id, 0) + 1

                if record.get("type") == "vitals":
                    if "breathing_bpm" in record and "heartrate_bpm" in record:
                        hw_vitals["breathing_bpm"].append(record["breathing_bpm"])
                        hw_vitals["heartrate_bpm"].append(record["heartrate_bpm"])
                        hw_vitals["motion_energy"].append(record.get("motion_energy", 0.0))
                        hw_vitals["timestamp"].append(record.get("timestamp", 0.0))
                    continue

                if "iq_hex" in record and record.get("type", "raw_csi") in ["raw_csi", None]:
                    if target_node is not None and node_id != target_node:
                        continue
                    subc = record.get("subcarriers", 64)
                    if self.n_subcarriers is None or len(frames_csi) == 0:
                        self.n_subcarriers = subc

                    if subc == self.n_subcarriers:
                        iq_hex = record["iq_hex"]
                        try:
                            raw_bytes = bytes.fromhex(iq_hex)
                            i_val = np.frombuffer(raw_bytes[0::2], dtype=np.int8).astype(np.float64)
                            q_val = np.frombuffer(raw_bytes[1::2], dtype=np.int8).astype(np.float64)
                            if len(i_val) == self.n_subcarriers:
                                complex_slice = i_val + 1j * q_val
                                frames_csi.append(complex_slice)
                                timestamps.append(record.get("timestamp", 0.0))
                                rssi_vals.append(record.get("rssi", -50))
                        except Exception:
                            continue

                if len(frames_csi) >= max_frames:
                    break

        if not frames_csi:
            raise ValueError(f"No valid CSI frames extracted from {file_path} for subcarriers={self.n_subcarriers}")

        csi_matrix = np.array(frames_csi).T # Shape: (n_subcarriers, n_frames)
        n_frames = csi_matrix.shape[1]
        csi_3d = np.zeros((self.n_subcarriers, n_frames, self.n_antennas), dtype=np.complex128)
        for a in range(self.n_antennas):
            phase_shift = np.exp(-1j * np.pi * a * np.sin(np.radians(15.0)))
            csi_3d[:, :, a] = csi_matrix * phase_shift

        ts_arr = np.array(timestamps)
        if len(ts_arr) > 1:
            duration = ts_arr[-1] - ts_arr[0]
            if duration > 0:
                self.sample_rate_hz = float(len(ts_arr) / duration)

        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "n_frames": n_frames,
            "duration_secs": float(ts_arr[-1] - ts_arr[0]) if len(ts_arr) > 1 else 0.0,
            "sample_rate_hz": self.sample_rate_hz,
            "mean_rssi_dbm": float(np.mean(rssi_vals)) if rssi_vals else -50.0,
            "node_counts": node_counts,
            "hw_vitals": hw_vitals
        }
        return csi_3d, ts_arr, metadata

    def remove_outliers_hampel(self, csi_matrix: np.ndarray, window_size: int = 5, threshold: float = 3.0) -> np.ndarray:
        """
        Apply a Hampel filter (MAD-based outlier removal) along the time axis (axis 1)
        to remove AGC gain spikes and RF interference bursts.
        
        Args:
            csi_matrix: Complex array H(f, t, a) of shape (N_f, N_t, N_a).
            window_size: Half-window size for local median estimation across time t.
            threshold: Number of MAD deviations to trigger outlier replacement.
            
        Returns:
            Cleaned complex array of the same shape.
        """
        n_subcarriers, n_packets, n_antennas = csi_matrix.shape
        cleaned = csi_matrix.copy()
        
        # We filter amplitude and phase separately or filter complex real/imag parts.
        # Operating on real and imaginary components preserves complex phase integrity.
        for is_real in [True, False]:
            comp = np.real(csi_matrix) if is_real else np.imag(csi_matrix)
            for t in range(n_packets):
                t_start = max(0, t - window_size)
                t_end = min(n_packets, t + window_size + 1)
                
                window_slice = comp[:, t_start:t_end, :]
                local_median = np.median(window_slice, axis=1, keepdims=True)
                mad = np.median(np.abs(window_slice - local_median), axis=1, keepdims=True)
                
                # Prevent division by zero in constant signals
                mad = np.maximum(mad, 1e-10)
                
                # Identify spikes at time t
                current_val = comp[:, t:t+1, :]
                dev = np.abs(current_val - local_median)
                outlier_mask = (dev / (1.4826 * mad)) > threshold
                
                # Replace outliers with local median in the cleaned complex matrix
                if np.any(outlier_mask):
                    outlier_idx = np.where(outlier_mask[:, 0, :])
                    if is_real:
                        cleaned.real[outlier_idx[0], t, outlier_idx[1]] = local_median[outlier_idx[0], 0, outlier_idx[1]]
                    else:
                        cleaned.imag[outlier_idx[0], t, outlier_idx[1]] = local_median[outlier_idx[0], 0, outlier_idx[1]]
        return cleaned

    def apply_lowpass_filter(self, csi_matrix: np.ndarray, cutoff_freq_hz: float = 15.0, order: int = 4) -> np.ndarray:
        """
        Apply a zero-phase digital Butterworth low-pass filter along the time axis
        to suppress thermal noise while preserving structural reflection Doppler dynamics.
        
        Args:
            csi_matrix: Complex array H(f, t, a) of shape (N_f, N_t, N_a).
            cutoff_freq_hz: Low-pass filter cutoff frequency in Hz.
            order: Order of the Butterworth filter.
            
        Returns:
            Filtered complex array H(f, t, a).
        """
        nyquist = 0.5 * self.sample_rate_hz
        normal_cutoff = cutoff_freq_hz / nyquist
        
        # Ensure cutoff is within valid filter bounds (0, 1)
        normal_cutoff = np.clip(normal_cutoff, 0.01, 0.99)
        
        b, a_coeffs = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        # Apply zero-phase forward-backward filtering across time axis (axis 1)
        filtered_real = signal.filtfilt(b, a_coeffs, np.real(csi_matrix), axis=1)
        filtered_imag = signal.filtfilt(b, a_coeffs, np.imag(csi_matrix), axis=1)
        
        return filtered_real + 1j * filtered_imag

    def interpolate_dropped_packets(self, csi_matrix: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect packet drops by analyzing time gaps between consecutive frames.
        Linearly interpolate missing frames on a uniform time grid.
        
        Args:
            csi_matrix: Complex array H(f, t, a) of shape (N_f, N_t, N_a).
            timestamps: Monotonically increasing timestamps in seconds (length N_t).
            
        Returns:
            Tuple of (interpolated_csi_matrix, uniform_timestamps).
        """
        expected_dt = 1.0 / self.sample_rate_hz
        t_start = timestamps[0]
        t_end = timestamps[-1]
        
        # Generate uniform time grid based on expected sampling frequency
        num_uniform_steps = int(np.round((t_end - t_start) / expected_dt)) + 1
        uniform_timestamps = np.linspace(t_start, t_end, num_uniform_steps)
        
        n_subcarriers, _, n_antennas = csi_matrix.shape
        uniform_csi = np.zeros((n_subcarriers, num_uniform_steps, n_antennas), dtype=np.complex128)
        
        # Interpolate real and imaginary components separately for each subcarrier & antenna
        for f in range(n_subcarriers):
            for a in range(n_antennas):
                real_interp = np.interp(uniform_timestamps, timestamps, np.real(csi_matrix[f, :, a]))
                imag_interp = np.interp(uniform_timestamps, timestamps, np.imag(csi_matrix[f, :, a]))
                uniform_csi[f, :, a] = real_interp + 1j * imag_interp
                
        return uniform_csi, uniform_timestamps

    def process_pipeline(self, csi_matrix: np.ndarray, timestamps: Optional[np.ndarray] = None,
                         apply_hampel: bool = True, apply_lpf: bool = True,
                         cutoff_freq_hz: float = 15.0) -> np.ndarray:
        """
        Execute the complete ingestion cleaning pipeline:
        1. Packet drop interpolation (if timestamps provided)
        2. Hampel outlier spike rejection
        3. Butterworth zero-phase low-pass filtering
        
        Args:
            csi_matrix: Raw complex array H(f, t, a).
            timestamps: Optional array of timestamps for drop interpolation.
            apply_hampel: Boolean flag to enable Hampel spike removal.
            apply_lpf: Boolean flag to enable Butterworth low-pass filtering.
            cutoff_freq_hz: Low pass cutoff frequency in Hz.
            
        Returns:
            Cleaned and uniformly sampled CSI matrix H(f, t, a).
        """
        processed_csi = csi_matrix.copy()
        
        if timestamps is not None and len(timestamps) == csi_matrix.shape[1]:
            expected_dt = 1.0 / self.sample_rate_hz
            if len(timestamps) > 1 and np.max(np.diff(timestamps)) > 1.5 * expected_dt:
                processed_csi, _ = self.interpolate_dropped_packets(processed_csi, timestamps)
            
        if apply_hampel:
            processed_csi = self.remove_outliers_hampel(processed_csi, window_size=5, threshold=3.0)
            
        if apply_lpf and processed_csi.shape[1] > 12: # Ensure sufficient samples for filtfilt order
            processed_csi = self.apply_lowpass_filter(processed_csi, cutoff_freq_hz=cutoff_freq_hz, order=4)
            
        return processed_csi

if __name__ == "__main__":
    # Self-test unit verification
    print("[*] Verifying CSIIngestionEngine module...")
    engine = CSIIngestionEngine(n_subcarriers=64, n_antennas=4, sample_rate_hz=100.0)
    
    # Synthesize noisy test matrix with an impulsive outlier spike
    t_grid = np.linspace(0, 1.0, 100)
    clean_signal = np.exp(-1j * 2 * np.pi * 5.0 * t_grid) # 5 Hz Doppler sinusoid
    test_csi = np.tile(clean_signal, (64, 4, 1)).transpose((0, 2, 1))
    test_csi += 0.1 * (np.random.randn(*test_csi.shape) + 1j * np.random.randn(*test_csi.shape))
    
    # Inject large outlier at time index 50
    test_csi[10, 50, 0] += 50.0 + 50.0j
    
    cleaned_csi = engine.process_pipeline(test_csi, timestamps=t_grid, apply_hampel=True, apply_lpf=True)
    
    spike_before = np.abs(test_csi[10, 50, 0])
    spike_after = np.abs(cleaned_csi[10, 50, 0])
    print(f"[+] Outlier attenuation: before={spike_before:.2f}, after={spike_after:.2f}")
    assert spike_after < 5.0, "Hampel filter failed to attenuate impulsive outlier spike!"
    print("[+] CSIIngestionEngine verification passed successfully!")
