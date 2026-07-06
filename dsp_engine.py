#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: DSP & Signal Processing Engine
Wi-Fi CSI Indoor Physical Geometry Reconstruction & Vital Sign Tracking Pipeline

This module implements:
1. Phase Sanitization: Linear regression slope removal for SFO/CFO calibration.
2. Background Subtraction: Rolling average removal of static multipath.
3. Power Delay Profile (PDP): Windowed IFFT across subcarriers.
4. 2D-MUSIC Algorithm: Joint Time-of-Flight (ToF) and Angle-of-Arrival (AoA) estimation.
5. Material-Aware Semantic Classifier: Reconciling reflection attenuation with
   Free Space Path Loss (FSPL) against ITU-R P.1238 / Keenetic attenuation benchmarks.
6. Dynamic Human Tracking & Vital Sign Extraction: Multi-band Butterworth IIR
   filtering (Respiration 0.1-0.4 Hz, Heart Rate 0.8-2.0 Hz) and FFT peak extraction.

Physics & Math Formulation:
- Phase Slope Sanitization: SFO and CFO induce a linear phase gradient across
  subcarrier frequencies f. For unwrapped phase phi(f):
      phi_linear(f) = alpha * f + beta
      phi_sanitized(f) = phi(f) - phi_linear(f)
- Friis FSPL & Material Residual Loss: Total attenuation is composed of propagation
  spreading loss and material interaction loss:
      FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi / c)
      Residual_Loss = Raw_CSI_Attenuation - FSPL
- Doppler Vital Modulation: Micro-displacements of the chest wall delta_x(t)
  modulate the reflected carrier phase: delta_phi(t) = (4*pi / lambda) * delta_x(t).
"""

import numpy as np
from scipy import signal, fft, linalg, stats
from typing import Tuple, Dict, List, Optional, Union, Any

# ==============================================================================
# ATTENUATION DATABASE (ITU-R P.1238 & Keenetic Benchmarks)
# dB loss coefficients at 2.4 GHz and 5.0/5.2 GHz
# ==============================================================================
ATTENUATION_DATABASE_DB = {
    'drywall':             {'2.4GHz': 1.5,  '5GHz': 3.0},
    'wood_door':           {'2.4GHz': 2.5,  '5GHz': 4.5},
    'glass':               {'2.4GHz': 2.0,  '5GHz': 3.5},
    'brick':               {'2.4GHz': 8.0,  '5GHz': 12.0},
    'reinforced_concrete': {'2.4GHz': 15.0, '5GHz': 25.0},
    'metal':               {'2.4GHz': 28.0, '5GHz': 30.0}
}

# Material color coding for Matplotlib visualization
MATERIAL_COLORS = {
    'drywall':             '#FFA500', # Orange / Yellow
    'wood_door':           '#8B4513', # Saddle Brown
    'glass':               '#00CED1', # Dark Turquoise / Blue
    'brick':               '#B22222', # Firebrick Red
    'reinforced_concrete': '#708090', # Slate Grey
    'metal':               '#FF0000', # Red / Crimson
    'unknown':             '#000000'  # Black
}

class CSIDSPEngine:
    """
    Digital Signal Processing engine for Wi-Fi CSI spatial geometry reconstruction
    and dynamic vital sign extraction.
    """
    def __init__(self, carrier_freq_hz: float = 5.2e9, bandwidth_hz: float = 40.0e6,
                 n_subcarriers: int = 64, n_antennas: int = 4, sample_rate_hz: float = 100.0):
        """
        Initialize DSP engine parameters.
        
        Args:
            carrier_freq_hz: Active carrier frequency in Hz (default 5200 MHz / 5.2 GHz).
            bandwidth_hz: Wi-Fi channel bandwidth in Hz (default 40 MHz).
            n_subcarriers: Number of OFDM subcarriers.
            n_antennas: Number of ULA receive antennas.
            sample_rate_hz: CSI packet sampling frequency in Hz.
        """
        self.fc = carrier_freq_hz
        self.bw = bandwidth_hz
        self.n_sub = n_subcarriers
        self.n_ant = n_antennas
        self.fs = sample_rate_hz
        self.c = 299792458.0 # Speed of light in m/s
        self.wavelength = self.c / self.fc
        
        # Subcarrier frequency grid relative to center frequency
        self.subcarrier_freqs = np.linspace(-self.bw / 2.0, self.bw / 2.0, self.n_sub)
        self.abs_freqs = self.fc + self.subcarrier_freqs

    def sanitize_phase_sfo_cfo(self, csi_matrix: np.ndarray) -> np.ndarray:
        """
        Remove linear phase slopes caused by Sampling Frequency Offset (SFO) and
        Carrier Frequency Offset (CFO) using linear regression across subcarriers.
        
        Args:
            csi_matrix: Complex array H(f, t, a) of shape (N_f, N_t, N_a).
            
        Returns:
            Sanitized complex matrix of the same shape with true structural phase.
        """
        n_sub, n_pkt, n_ant = csi_matrix.shape
        sanitized = np.zeros_like(csi_matrix, dtype=np.complex128)
        
        # Linear regression matrix X for subcarrier indices [0, 1, ..., N_sub-1]
        x_idx = np.arange(n_sub)
        x_mean = np.mean(x_idx)
        x_var = np.var(x_idx)
        
        for t in range(n_pkt):
            for a in range(n_ant):
                raw_complex = csi_matrix[:, t, a]
                amplitude = np.abs(raw_complex)
                
                # Unwrap phase across frequency subcarriers
                unwrapped_phase = np.unwrap(np.angle(raw_complex))
                
                # Fit linear slope alpha and offset beta
                y_mean = np.mean(unwrapped_phase)
                cov_xy = np.mean((x_idx - x_mean) * (unwrapped_phase - y_mean))
                alpha = cov_xy / np.maximum(x_var, 1e-10)
                beta = y_mean - alpha * x_mean
                
                # Subtract linear slope induced by packet detection delay / SFO / CFO
                linear_phase = alpha * x_idx + beta
                clean_phase = unwrapped_phase - linear_phase
                
                sanitized[:, t, a] = amplitude * np.exp(1j * clean_phase)
                
        return sanitized

    def remove_static_background(self, csi_matrix: np.ndarray, rolling_window: int = 500) -> np.ndarray:
        """
        Isolate dynamic reflections (human movement/breathing) by subtracting a rolling
        moving average of the static room environment across the last N packets.
        
        Args:
            csi_matrix: Complex array H(f, t, a) of shape (N_f, N_t, N_a).
            rolling_window: Number of historical packets to include in static average.
            
        Returns:
            Dynamic variation matrix H_dynamic(f, t, a).
        """
        n_sub, n_pkt, n_ant = csi_matrix.shape
        dynamic_csi = np.zeros_like(csi_matrix, dtype=np.complex128)
        
        for t in range(n_pkt):
            t_start = max(0, t - rolling_window)
            static_bg = np.mean(csi_matrix[:, t_start:t+1, :], axis=1)
            dynamic_csi[:, t, :] = csi_matrix[:, t, :] - static_bg
            
        return dynamic_csi

    def compute_pdp_ifft(self, csi_matrix: np.ndarray, window_type: str = 'hamming',
                         n_fft_pad: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Power Delay Profile (PDP) using windowed Inverse Fast Fourier
        Transform across OFDM subcarriers.
        
        Args:
            csi_matrix: Complex array H(f, t, a) of shape (N_f, N_t, N_a).
            window_type: Spectral windowing function ('hamming', 'blackman', 'hann').
            n_fft_pad: Zero-padded IFFT size for high-resolution time interpolation.
            
        Returns:
            Tuple of (pdp_matrix, delay_axis_seconds), where pdp_matrix has shape
            (n_fft_pad, N_t, N_a).
        """
        n_sub, n_pkt, n_ant = csi_matrix.shape
        
        # Construct frequency window
        if window_type == 'hamming':
            win = np.hamming(n_sub)
        elif window_type == 'blackman':
            win = np.blackman(n_sub)
        else:
            win = np.hanning(n_sub)
            
        win_matrix = win[:, np.newaxis, np.newaxis]
        windowed_csi = csi_matrix * win_matrix
        
        # Apply IFFT across axis 0 (subcarriers) with zero padding
        ifft_result = fft.ifft(windowed_csi, n=n_fft_pad, axis=0)
        pdp_matrix = np.abs(ifft_result)**2
        
        # Time delay resolution delta_tau = 1 / Bandwidth
        delay_step = 1.0 / (self.bw * (n_fft_pad / n_sub))
        delay_axis = np.arange(n_fft_pad) * delay_step
        
        return pdp_matrix, delay_axis

    def estimate_2d_music(self, csi_snapshot: np.ndarray, n_sources: int = 4,
                          angle_grid_deg: np.ndarray = np.linspace(-60, 60, 121),
                          delay_grid_ns: np.ndarray = np.linspace(0, 50, 101)) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
        """
        High-resolution 2D-MUSIC algorithm for joint Time-of-Flight (ToF) and
        Angle-of-Arrival (AoA) estimation from a subcarrier-antenna matrix snapshot.
        
        Args:
            csi_snapshot: Complex matrix H(f, a) of shape (N_sub, N_ant) at time t.
            n_sources: Number of discrete reflection paths to isolate (LoS + walls).
            angle_grid_deg: Search grid of AoA angles in degrees.
            delay_grid_ns: Search grid of ToF delays in nanoseconds.
            
        Returns:
            Tuple of (pseudospectrum_2d, list_of_peaks_deg_ns_val).
        """
        n_sub, n_ant = csi_snapshot.shape
        
        # We apply spatial-temporal smoothing by partitioning into overlapping sub-arrays
        sub_f = min(16, n_sub // 2)
        sub_a = min(3, n_ant) if n_ant >= 3 else n_ant
        
        num_f_steps = n_sub - sub_f + 1
        num_a_steps = n_ant - sub_a + 1
        
        # Assemble smoothed covariance matrix Rxx
        vec_len = sub_f * sub_a
        r_xx = np.zeros((vec_len, vec_len), dtype=np.complex128)
        
        for k_f in range(num_f_steps):
            for k_a in range(num_a_steps):
                sub_mat = csi_snapshot[k_f:k_f+sub_f, k_a:k_a+sub_a]
                vec = sub_mat.flatten()[:, np.newaxis]
                r_xx += vec @ vec.conj().T
                
        r_xx /= (num_f_steps * num_a_steps)
        
        # Eigenvalue Decomposition (EVD) of Rxx
        evals, evecs = linalg.eigh(r_xx)
        
        # Sort eigenvalues in descending order
        idx_sort = np.argsort(evals)[::-1]
        evecs = evecs[:, idx_sort]
        
        # Noise subspace En corresponds to the smallest eigenvalues
        n_sources_eff = min(n_sources, vec_len - 1)
        e_noise = evecs[:, n_sources_eff:]
        en_en_h = e_noise @ e_noise.conj().T
        
        # Precompute sub-array frequency and antenna spatial frequencies
        f_sub_grid = self.subcarrier_freqs[:sub_f]
        ant_idx = np.arange(sub_a)
        
        pseudospectrum = np.zeros((len(angle_grid_deg), len(delay_grid_ns)), dtype=np.float64)
        
        for idx_theta, theta_deg in enumerate(angle_grid_deg):
            theta_rad = np.radians(theta_deg)
            # Spatial steering vector along antenna array (ULA with lambda/2 spacing)
            spatial_phase = -np.pi * np.sin(theta_rad) * ant_idx
            a_spatial = np.exp(1j * spatial_phase)
            
            for idx_tau, tau_ns in enumerate(delay_grid_ns):
                tau_sec = tau_ns * 1e-9
                # Temporal frequency steering vector
                temporal_phase = -2.0 * np.pi * f_sub_grid * tau_sec
                a_temporal = np.exp(1j * temporal_phase)
                
                # 2D Kronecker product steering vector
                a_2d = np.kron(a_spatial, a_temporal)[:, np.newaxis]
                
                # MUSIC Pseudospectrum metric: 1 / (a^H * En * En^H * a)
                denom = np.real(a_2d.conj().T @ en_en_h @ a_2d)[0, 0]
                pseudospectrum[idx_theta, idx_tau] = 1.0 / np.maximum(denom, 1e-12)
                
        # Normalize pseudospectrum
        pseudospectrum /= np.max(pseudospectrum)
        
        # Find 2D local peaks representing multipath reflections
        peaks = []
        flat_idx = np.argsort(pseudospectrum.flatten())[::-1]
        for idx in flat_idx:
            if len(peaks) >= n_sources:
                break
            th_i, tau_i = np.unravel_index(idx, pseudospectrum.shape)
            th_val = angle_grid_deg[th_i]
            tau_val = delay_grid_ns[tau_i]
            p_val = pseudospectrum[th_i, tau_i]
            
            # Check spacing from existing peaks to avoid duplicates within beamwidth
            is_distinct = True
            for exist_th, exist_tau, _ in peaks:
                if np.hypot(th_val - exist_th, (tau_val - exist_tau) * 2.0) < 10.0:
                    is_distinct = False
                    break
            if is_distinct and p_val > 0.05:
                peaks.append((float(th_val), float(tau_val), float(p_val)))
                
        return pseudospectrum, peaks

    def classify_material_semantic(self, raw_attenuation_db: float, tof_ns: float,
                                   freq_band: str = '5GHz') -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Reconcile reflection attenuation with geometric Free Space Path Loss (FSPL)
        and execute a Bayesian probability classification against ITU-R P.1238 / Keenetic.
        
        Args:
            raw_attenuation_db: Total observed attenuation of the reflection in dB.
            tof_ns: Time-of-Flight delay in nanoseconds.
            freq_band: Frequency band key ('5GHz' or '2.4GHz').
            
        Returns:
            Dictionary containing predicted material label, confidence probability,
            calculated FSPL, residual loss, and full probability distribution.
        """
        # Calculate propagation distance from ToF
        distance_m = (tof_ns * 1e-9) * self.c
        distance_m = np.maximum(distance_m, 0.1) # Prevent log(0)
        
        # Calculate Free Space Path Loss (Friis formula in dB)
        fspl_db = 20.0 * np.log10(distance_m) + 20.0 * np.log10(self.fc) + 20.0 * np.log10(4.0 * np.pi / self.c)
        
        # Isolate material interaction residual loss
        residual_loss_db = raw_attenuation_db - fspl_db
        
        # We model each material's expected loss with a Gaussian likelihood (sigma = 2.5 dB)
        sigma_db = 2.5
        log_likelihoods = {}
        total_prob = 0.0
        
        for mat_name, constants in ATTENUATION_DATABASE_DB.items():
            expected_loss = constants.get(freq_band, constants['5GHz'])
            # Gaussian unnormalized likelihood
            prob = np.exp(-0.5 * ((residual_loss_db - expected_loss) / sigma_db)**2)
            log_likelihoods[mat_name] = float(prob)
            total_prob += prob
            
        # Normalize into Bayesian posterior probabilities
        prob_dist = {}
        best_mat = 'unknown'
        max_prob = -1.0
        
        for mat_name, prob in log_likelihoods.items():
            norm_p = prob / np.maximum(total_prob, 1e-12)
            prob_dist[mat_name] = float(norm_p)
            if norm_p > max_prob:
                max_prob = norm_p
                best_mat = mat_name
                
        return {
            'predicted_material': best_mat,
            'confidence': max_prob,
            'fspl_db': float(fspl_db),
            'residual_loss_db': float(residual_loss_db),
            'probabilities': prob_dist,
            'color': MATERIAL_COLORS.get(best_mat, '#000000'),
            'label_str': f"Wall Segment: {int(max_prob * 100)}% Probability {best_mat.replace('_', ' ').title()}"
        }

    def filter_vital_signs(self, phase_time_series: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """
        Multi-band Butterworth IIR filtering and FFT spectral analysis to extract
        respiration waveforms (0.1-0.4 Hz) and heart rates (0.8-2.0 Hz) from rolling CSI phase.
        
        Args:
            phase_time_series: Unwrapped phase trajectory across time t (length N_t).
            
        Returns:
            Dictionary containing respiration waveform, heart rate waveform,
            extracted BPM (breaths/min), extracted HR (beats/min), and FFT spectrum.
        """
        n_samples = len(phase_time_series)
        nyquist = 0.5 * self.fs
        
        # Ensure minimum samples for filtfilt
        if n_samples < 30:
            return {
                'resp_wave': np.zeros(n_samples),
                'hr_wave': np.zeros(n_samples),
                'bpm_resp': 0.0,
                'hr_bpm': 0.0,
                'fft_freqs': np.linspace(0, 5, 50),
                'fft_hr_mag': np.zeros(50)
            }
            
        # 1. Respiration Filter: Bandpass between 0.1 Hz and 0.4 Hz (6 to 24 breaths/min)
        low_resp = np.clip(0.1 / nyquist, 0.0001, 0.9998)
        high_resp = np.clip(0.4 / nyquist, low_resp + 0.0001, 0.9999)
        b_resp, a_resp = signal.butter(3, [low_resp, high_resp], btype='bandpass')
        resp_wave = signal.filtfilt(b_resp, a_resp, phase_time_series)
        
        # 2. Heart Rate Filter: Bandpass between 0.8 Hz and 2.0 Hz (48 to 120 beats/min)
        low_hr = np.clip(0.8 / nyquist, 0.0001, 0.9998)
        high_hr = np.clip(2.0 / nyquist, low_hr + 0.0001, 0.9999)
        b_hr, a_hr = signal.butter(3, [low_hr, high_hr], btype='bandpass')
        hr_wave = signal.filtfilt(b_hr, a_hr, phase_time_series)
        
        # FFT analysis on rolling window (apply Hanning window to reduce leakage)
        win = np.hanning(n_samples)
        
        # Respiration FFT
        fft_resp = np.abs(fft.rfft(resp_wave * win))
        freqs_resp = fft.rfftfreq(n_samples, d=1.0/self.fs)
        valid_resp_idx = np.where((freqs_resp >= 0.1) & (freqs_resp <= 0.4))[0]
        if len(valid_resp_idx) > 0:
            best_idx = valid_resp_idx[np.argmax(fft_resp[valid_resp_idx])]
            bpm_resp = float(freqs_resp[best_idx] * 60.0)
        else:
            bpm_resp = 16.0 # Default fallback nominal breathing rate
            
        # Heart Rate FFT
        fft_hr = np.abs(fft.rfft(hr_wave * win))
        freqs_hr = fft.rfftfreq(n_samples, d=1.0/self.fs)
        valid_hr_idx = np.where((freqs_hr >= 0.8) & (freqs_hr <= 2.0))[0]
        if len(valid_hr_idx) > 0:
            best_hr_idx = valid_hr_idx[np.argmax(fft_hr[valid_hr_idx])]
            hr_bpm = float(freqs_hr[best_hr_idx] * 60.0)
        else:
            hr_bpm = 72.0 # Default fallback nominal heart rate
            
        return {
            'resp_wave': resp_wave,
            'hr_wave': hr_wave,
            'bpm_resp': bpm_resp,
            'hr_bpm': hr_bpm,
            'fft_freqs': freqs_hr,
            'fft_hr_mag': fft_hr
        }

if __name__ == "__main__":
    print("[*] Verifying CSIDSPEngine module...")
    dsp = CSIDSPEngine(carrier_freq_hz=5.2e9, bandwidth_hz=40.0e6, n_subcarriers=64, n_antennas=4, sample_rate_hz=100.0)
    
    # Verify phase sanitization
    t_idx = np.arange(100)
    raw_csi = np.ones((64, 100, 4), dtype=np.complex128)
    # Inject linear slope across subcarriers (simulating SFO/PDD)
    slope = 0.1
    for f in range(64):
        raw_csi[f, :, :] *= np.exp(1j * slope * f)
        
    sanitized = dsp.sanitize_phase_sfo_cfo(raw_csi)
    res_phase = np.mean(np.abs(np.angle(sanitized)))
    print(f"[+] SFO phase slope residual after sanitization: {res_phase:.6f} rad")
    assert res_phase < 1e-4, "Phase sanitization failed to eliminate linear phase slope!"
    
    # Verify Material Classifier
    mat_result = dsp.classify_material_semantic(raw_attenuation_db=78.5, tof_ns=15.0, freq_band='5GHz')
    print(f"[+] Material Classifier Result: {mat_result['label_str']} (Residual Loss: {mat_result['residual_loss_db']:.1f} dB)")
    
    # Verify Vital Sign Filter
    t_sec = np.linspace(0, 15.0, 1500) # 15 seconds at 100 Hz
    # Synthesize 0.25 Hz respiration (15 BPM) + 1.2 Hz heartbeat (72 BPM)
    sim_phase = 0.5 * np.sin(2 * np.pi * 0.25 * t_sec) + 0.05 * np.sin(2 * np.pi * 1.2 * t_sec)
    sim_phase += 0.02 * np.random.randn(len(t_sec))
    
    vitals = dsp.filter_vital_signs(sim_phase)
    print(f"[+] Extracted Vital Signs: Respiration = {vitals['bpm_resp']:.1f} BPM, Heart Rate = {vitals['hr_bpm']:.1f} BPM")
    assert abs(vitals['bpm_resp'] - 15.0) < 2.0, f"Respiration rate estimation error: {vitals['bpm_resp']}"
    assert abs(vitals['hr_bpm'] - 72.0) < 5.0, f"Heart rate estimation error: {vitals['hr_bpm']}"
    
    print("[+] CSIDSPEngine verification passed successfully!")
