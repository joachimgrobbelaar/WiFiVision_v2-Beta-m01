#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: Geometric Mapping & Reconstruction
Wi-Fi CSI Indoor Physical Geometry Reconstruction & Dynamic Doppler Tracking

This module implements:
1. Coordinate Transformation: Converting Time-of-Flight (ToF) and Angle-of-Arrival (AoA)
   multipath vectors into 2D Cartesian boundary reflection coordinates.
2. DBSCAN Clustering & Semantic Wall Attribution: Grouping reflection point clouds into
   distinct structural clusters, fitting continuous linear wall segments via SVD/PCA,
   and assigning semantic material labels via ITU-R P.1238 / Keenetic reconciliation.
3. Dynamic Doppler Macro-Tracking: Tracing the spatial coordinate path of a walking
   human target by extracting dynamic multipath delays from background-subtracted CSI.

Physics & Math Formulation:
- Monostatic Virtual Source Mapping: For a transceiver located at origin (x0, y0),
  a reflection with two-way propagation delay tau and angle theta corresponds to a
  reflecting boundary coordinate at distance r = (c * tau) / 2:
      x = x0 + r * cos(theta),    y = y0 + r * sin(theta)
- PCA Line Segment Fitting: For a cluster of points P_c = [x_i, y_i], the wall boundary
  orientation is given by the principal eigenvector v1 of the covariance matrix Cov(P_c).
- Doppler Velocity & Path Tracking: A target walking at velocity v generates Doppler shift
  fd = (2*v*fc/c)*cos(psi). The time-varying delay tau_dyn(t) tracks the target trajectory.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy import ndimage
from typing import List, Dict, Tuple, Optional, Any
from dsp_engine import CSIDSPEngine

class GeometricMapper:
    """
    Geometric mapping engine for room boundary reconstruction and dynamic target tracking.
    """
    def __init__(self, dsp_engine: CSIDSPEngine, transceiver_origin: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize the mapper.
        
        Args:
            dsp_engine: Instance of CSIDSPEngine for material classification and physics constants.
            transceiver_origin: Spatial (x, y) origin coordinates of the transceiver array in meters.
        """
        self.dsp = dsp_engine
        self.origin_x, self.origin_y = transceiver_origin

    def map_peaks_to_cartesian(self, peaks: List[Tuple[float, float, float]],
                               tof_offset_ns: float = 0.0) -> np.ndarray:
        """
        Convert polar ToF delay (ns) and AoA angle (deg) peaks to 2D Cartesian
        coordinates (x, y) relative to the transceiver origin.
        
        Args:
            peaks: List of (aoa_deg, tof_ns, peak_power) tuples from 2D-MUSIC.
            tof_offset_ns: System hardware delay offset in nanoseconds.
            
        Returns:
            np.ndarray of shape (N_peaks, 3) containing [x_m, y_m, peak_power].
        """
        if not peaks:
            return np.empty((0, 3))
            
        coords = []
        for aoa_deg, tof_ns, power in peaks:
            # Correct for hardware delay offset
            effective_tof_ns = max(0.1, tof_ns - tof_offset_ns)
            
            # Calculate total propagation distance and radial reflection distance
            total_dist_m = (effective_tof_ns * 1e-9) * self.dsp.c
            radial_dist_m = total_dist_m / 2.0
            
            # Convert polar angle to radians (standard Cartesian convention)
            aoa_rad = np.radians(aoa_deg)
            
            x_m = self.origin_x + radial_dist_m * np.cos(aoa_rad)
            y_m = self.origin_y + radial_dist_m * np.sin(aoa_rad)
            coords.append([x_m, y_m, power])
            
        return np.array(coords)

    def cluster_and_reconstruct_walls(self, points_2d: np.ndarray, residual_losses_db: np.ndarray,
                                      eps_m: float = 0.8, min_samples: int = 3,
                                      freq_band: str = '5GHz') -> List[Dict[str, Any]]:
        """
        Apply DBSCAN density clustering to aggregate noisy reflection point clouds,
        fit continuous linear wall segments via PCA/SVD, and classify wall materials.
        
        Args:
            points_2d: Array of reflection points (N, 2) or (N, 3).
            residual_losses_db: Array of observed attenuation/residual loss for each point.
            eps_m: DBSCAN spatial neighborhood search radius in meters.
            min_samples: Minimum points required to form a valid structural cluster.
            freq_band: Frequency band key for material classification ('5GHz' or '2.4GHz').
            
        Returns:
            List of dictionary structures representing reconstructed wall segments with
            endpoints, centroid, material label, confidence, and color.
        """
        if len(points_2d) < min_samples:
            return []
            
        xy_data = points_2d[:, :2]
        dbscan = DBSCAN(eps=eps_m, min_samples=min_samples)
        labels = dbscan.fit_predict(xy_data)
        
        unique_labels = set(labels)
        wall_segments = []
        
        for cluster_id in unique_labels:
            if cluster_id == -1:
                # Skip DBSCAN noise outliers
                continue
                
            cluster_mask = (labels == cluster_id)
            cluster_pts = xy_data[cluster_mask]
            cluster_losses = residual_losses_db[cluster_mask]
            
            if len(cluster_pts) < 2:
                continue
                
            # Calculate centroid
            centroid = np.mean(cluster_pts, axis=0)
            
            # Fit line segment using Principal Component Analysis (PCA / SVD)
            centered_pts = cluster_pts - centroid
            u, s, vh = np.linalg.svd(centered_pts)
            principal_dir = vh[0] # Eigenvector corresponding to largest variance along wall
            
            # Project points onto principal direction to find wall segment endpoints
            projections = centered_pts @ principal_dir
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            
            endpoint_start = centroid + min_proj * principal_dir
            endpoint_end = centroid + max_proj * principal_dir
            
            # Compute mean residual loss for material classification
            mean_loss_db = float(np.mean(cluster_losses))
            mean_tof_ns = float((np.linalg.norm(centroid - np.array([self.origin_x, self.origin_y])) * 2.0 / self.dsp.c) * 1e9)
            
            # Query Semantic Material Classifier
            mat_info = self.dsp.classify_material_semantic(raw_attenuation_db=mean_loss_db,
                                                           tof_ns=mean_tof_ns, freq_band=freq_band)
            
            wall_segments.append({
                'cluster_id': int(cluster_id),
                'num_points': int(len(cluster_pts)),
                'centroid': centroid.tolist(),
                'endpoint_start': endpoint_start.tolist(),
                'endpoint_end': endpoint_end.tolist(),
                'points': cluster_pts.tolist(),
                'mean_loss_db': mean_loss_db,
                'material': mat_info['predicted_material'],
                'confidence': mat_info['confidence'],
                'color': mat_info['color'],
                'label_str': mat_info['label_str']
            })
            
        return wall_segments

    def track_dynamic_target_path(self, dynamic_csi: np.ndarray, timestamps: np.ndarray,
                                  window_packets: int = 20) -> np.ndarray:
        """
        Track the spatial 2D macro-movement trajectory (walking path) of a dynamic
        human target across time by extracting dominant multipath delays and angles
        from the background-subtracted dynamic CSI matrix.
        
        Args:
            dynamic_csi: Background-subtracted matrix H_dynamic(f, t, a).
            timestamps: Array of timestamps in seconds.
            window_packets: Sliding window size for short-time tracking.
            
        Returns:
            np.ndarray of shape (N_steps, 3) containing [timestamp, x_m, y_m] walking path coordinates.
        """
        n_sub, n_pkt, n_ant = dynamic_csi.shape
        step_size = max(1, window_packets // 2)
        num_steps = (n_pkt - window_packets) // step_size + 1
        
        if num_steps <= 0:
            return np.empty((0, 3))
            
        trajectory = []
        
        for k in range(num_steps):
            t_idx = k * step_size
            t_mid = timestamps[min(n_pkt - 1, t_idx + window_packets // 2)]
            
            # Take mean snapshot across time window
            win_slice = dynamic_csi[:, t_idx:t_idx+window_packets, :]
            snapshot = np.mean(np.abs(win_slice) * np.exp(1j * np.angle(win_slice)), axis=1)
            
            # Run 1-source 2D-MUSIC to locate dominant dynamic scatterer (the moving human!)
            _, peaks = self.dsp.estimate_2d_music(snapshot, n_sources=1,
                                                  angle_grid_deg=np.linspace(-60, 60, 61),
                                                  delay_grid_ns=np.linspace(2.0, 30.0, 57))
            if peaks:
                aoa_deg, tof_ns, _ = peaks[0]
                # Map to Cartesian
                dist_m = (tof_ns * 1e-9) * self.dsp.c / 2.0
                aoa_rad = np.radians(aoa_deg)
                x_m = self.origin_x + dist_m * np.cos(aoa_rad)
                y_m = self.origin_y + dist_m * np.sin(aoa_rad)
                trajectory.append([t_mid, x_m, y_m])
                
        return np.array(trajectory)

    def map_bistatic_peaks_to_3d(self, peaks_3d: List[Tuple[float, float, float, float]],
                                 tx_pos_3d: Tuple[float, float, float] = (3.5, 4.0, 2.5),
                                 rx_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
        """
        Map 3D bistatic Time-of-Flight (ToF) and 3D arrival angles (Azimuth & Elevation)
        to exact 3D Cartesian coordinates (X, Y, Z) of physical obstacles in the room.
        
        Uses exact analytical ray-ellipsoid intersection:
            ||P - P_tx|| + ||P - P_rx|| = L = tau * c
        
        Args:
            peaks_3d: List of tuples (azimuth_deg, elevation_deg, tof_ns, peak_power).
            tx_pos_3d: 3D Cartesian coordinates of the Sender (Router / Access Point).
            rx_pos_3d: 3D Cartesian coordinates of the Receiver (Client / Sensing Array).
            
        Returns:
            np.ndarray of shape (N_valid, 4) containing [X_m, Y_m, Z_m, power].
        """
        if not peaks_3d:
            return np.empty((0, 4))
            
        tx = np.array(tx_pos_3d, dtype=float)
        rx = np.array(rx_pos_3d, dtype=float)
        delta = rx - tx # Vector pointing from TX to RX
        delta_sq = float(np.dot(delta, delta))
        baseline_dist = float(np.sqrt(delta_sq))
        
        coords = []
        for az_deg, el_deg, tof_ns, power in peaks_3d:
            # Total bistatic path length
            L = (tof_ns * 1e-9) * self.dsp.c
            if L <= baseline_dist + 1e-3:
                # Direct LoS path or unphysical reflection (< baseline distance)
                continue
                
            az_rad = np.radians(az_deg)
            el_rad = np.radians(el_deg)
            
            # Unit arrival ray vector u pointing from RX toward reflecting obstacle
            ux = np.cos(el_rad) * np.cos(az_rad)
            uy = np.cos(el_rad) * np.sin(az_rad)
            uz = np.sin(el_rad)
            u = np.array([ux, uy, uz], dtype=float)
            
            # Exact analytical quadratic solution for radial distance r along u
            delta_dot_u = float(np.dot(delta, u))
            denom = 2.0 * (L + delta_dot_u)
            if abs(denom) < 1e-6:
                continue
                
            r = (L * L - delta_sq) / denom
            if r <= 0.05 or r > 20.0:
                continue
                
            p_obs = rx + r * u
            coords.append([p_obs[0], p_obs[1], p_obs[2], power])
            
        return np.array(coords)

    def cluster_3d_obstacles(self, points_3d: np.ndarray, eps_m: float = 0.6,
                             min_samples: int = 3) -> List[Dict[str, Any]]:
        """
        Group 3D obstacle reflection point clouds using 3D DBSCAN and fit bounding geometry
        via 3D Principal Component Analysis (SVD) to classify structural obstacle types.
        
        Args:
            points_3d: Array of 3D reflection points (N, 3) or (N, 4).
            eps_m: DBSCAN spatial clustering radius in meters.
            min_samples: Minimum points required to identify a physical obstacle.
            
        Returns:
            List of 3D obstacle dictionaries containing centroid, bounding box, axes, and semantic label.
        """
        if len(points_3d) < min_samples:
            return []
            
        xyz_data = points_3d[:, :3]
        dbscan = DBSCAN(eps=eps_m, min_samples=min_samples)
        labels = dbscan.fit_predict(xyz_data)
        
        unique_labels = set(labels)
        obstacles = []
        
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue # Skip noise
                
            cluster_mask = (labels == cluster_id)
            cluster_pts = xyz_data[cluster_mask]
            
            if len(cluster_pts) < 2:
                continue
                
            centroid = np.mean(cluster_pts, axis=0)
            centered_pts = cluster_pts - centroid
            u, s, vh = np.linalg.svd(centered_pts)
            
            # Bounding box extents along principal axes
            min_bounds = np.min(cluster_pts, axis=0)
            max_bounds = np.max(cluster_pts, axis=0)
            extents = max_bounds - min_bounds
            
            # Semantic 3D Obstacle Classification based on spatial geometry
            if extents[0] > 1.2 or extents[1] > 1.2:
                obs_type = "Boundary Wall / Room Partition"
                color = "#2b5c8f" # Steel blue
            elif centroid[2] > 2.2:
                obs_type = "Ceiling Structure / Overhead Fixture"
                color = "#8f2b8f" # Purple
            elif centroid[2] < 0.4:
                obs_type = "Floor Boundary / Low Object"
                color = "#8f5c2b" # Brown
            elif extents[2] > 1.0:
                obs_type = "Vertical Column / Support Pillar"
                color = "#d95f02" # Orange
            else:
                obs_type = "Room Furniture (Desk / Cabinet)"
                color = "#1b9e77" # Teal green
                
            obstacles.append({
                'cluster_id': int(cluster_id),
                'num_points': int(len(cluster_pts)),
                'centroid': centroid.tolist(),
                'min_bounds': min_bounds.tolist(),
                'max_bounds': max_bounds.tolist(),
                'extents': extents.tolist(),
                'principal_axes': vh.tolist(),
                'points': cluster_pts.tolist(),
                'type': obs_type,
                'color': color
            })
            
        return obstacles

    def reconstruct_surface_gaussian_splats(self, points: np.ndarray, grid_size: int = 200,
                                            extent_m: float = 5.0, sigma_init: float = 0.15,
                                            step_size: float = 0.05, max_steps: int = 40) -> Dict[str, Any]:
        """
        Treat each reflection point as a 2D Gaussian splat that continuously grows out in all directions
        over a refined surface grid. Growth continues until a splat meets another splat (or room boundary),
        halting growth in that direction only (Voronoi-bounded anisotropic region growing).
        """
        x_lin = np.linspace(-extent_m / 2.0, extent_m / 2.0, grid_size)
        y_lin = np.linspace(-extent_m / 2.0, extent_m / 2.0, grid_size)
        xx, yy = np.meshgrid(x_lin, y_lin)
        
        if len(points) == 0:
            return {
                "X": xx, "Y": yy,
                "Z_surface": np.zeros_like(xx),
                "collision_mask": np.zeros_like(xx, dtype=bool),
                "edge_map": np.zeros_like(xx)
            }
            
        pts_2d = points[:, :2]
        weights = points[:, 2] if points.shape[1] > 2 else np.ones(len(pts_2d))
        n_pts = len(pts_2d)
        
        # We model directional growth limits for each splat around 16 angular sectors (0 to 360 deg)
        n_sectors = 16
        sector_angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
        # R_limits[i, s] stores the maximum growth radius of splat i in sector s
        R_limits = np.full((n_pts, n_sectors), sigma_init, dtype=float)
        
        # Simulate continuous region growth
        for step in range(max_steps):
            R_limits += step_size
            # Check for wavefront collisions between pairwise splats
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    dx = pts_2d[j, 0] - pts_2d[i, 0]
                    dy = pts_2d[j, 1] - pts_2d[i, 1]
                    dist_ij = np.hypot(dx, dy)
                    if dist_ij < 1e-4:
                        continue
                    
                    # Find sector from i to j and j to i
                    angle_ij = np.mod(np.arctan2(dy, dx), 2 * np.pi)
                    angle_ji = np.mod(angle_ij + np.pi, 2 * np.pi)
                    
                    s_ij = int(np.argmin(np.abs(np.angle(np.exp(1j * (sector_angles - angle_ij))))))
                    s_ji = int(np.argmin(np.abs(np.angle(np.exp(1j * (sector_angles - angle_ji))))))
                    
                    # If expanding wavefronts collide, freeze growth along collision line!
                    if R_limits[i, s_ij] + R_limits[j, s_ji] >= dist_ij:
                        half_dist = dist_ij / 2.0
                        R_limits[i, s_ij] = min(R_limits[i, s_ij], half_dist)
                        R_limits[j, s_ji] = min(R_limits[j, s_ji], half_dist)
                        
        # Reconstruct continuous surface height field across pixel grid using collision-bounded splats
        Z_surface = np.zeros_like(xx, dtype=float)
        collision_mask = np.zeros_like(xx, dtype=bool)
        
        # For each pixel, accumulate contributions from nearby splats bounded by R_limits
        for i in range(n_pts):
            dx_grid = xx - pts_2d[i, 0]
            dy_grid = yy - pts_2d[i, 1]
            dist_grid = np.hypot(dx_grid, dy_grid)
            angle_grid = np.mod(np.arctan2(dy_grid, dx_grid), 2 * np.pi)
            
            # Map grid angles to sector indices
            s_grid = np.round((angle_grid / (2 * np.pi)) * n_sectors).astype(int) % n_sectors
            max_r = R_limits[i, s_grid]
            
            # Splat contributes within its collision boundary
            valid_mask = dist_grid <= (max_r * 1.5)
            sigma_eff = max_r * 0.45 # Effective width proportional to allowed boundary
            
            # 2D Gaussian profile
            gauss_contrib = weights[i] * np.exp(-0.5 * (dist_grid / np.maximum(sigma_eff, 1e-3)) ** 2)
            Z_surface += np.where(valid_mask, gauss_contrib, 0.0)
            
            # Mark collision borders where growth was truncated by neighbors
            collision_mask |= (dist_grid > max_r) & (dist_grid < max_r * 1.2) & (max_r < (sigma_init + max_steps * step_size * 0.9))
            
        # Edge Detection: Apply spatial gradient operators across splatted surface field
        edge_map = self.compute_edge_detection(Z_surface, method='sobel')
        
        return {
            "X": xx, "Y": yy,
            "Z_surface": Z_surface,
            "collision_mask": collision_mask,
            "edge_map": edge_map,
            "R_limits": R_limits
        }

    def compute_edge_detection(self, surface_grid: np.ndarray, method: str = 'sobel') -> np.ndarray:
        """
        Apply spatial edge detection across the Gaussian splatted surface grid
        to delineate structural wall and obstacle contours.
        
        Args:
            surface_grid: 2D array representing surface intensity / elevation.
            method: Gradient operator ('sobel', 'laplace', 'canny_approx').
            
        Returns:
            2D edge magnitude map of the same shape.
        """
        if method == 'sobel':
            grad_x = ndimage.sobel(surface_grid, axis=1)
            grad_y = ndimage.sobel(surface_grid, axis=0)
            edge_map = np.hypot(grad_x, grad_y)
        elif method == 'laplace':
            edge_map = np.abs(ndimage.laplace(surface_grid))
        else:
            smoothed = ndimage.gaussian_filter(surface_grid, sigma=1.0)
            grad_x = ndimage.sobel(smoothed, axis=1)
            grad_y = ndimage.sobel(smoothed, axis=0)
            edge_map = np.hypot(grad_x, grad_y)
            
        max_val = np.max(edge_map)
        if max_val > 0:
            edge_map /= max_val
        return edge_map

if __name__ == "__main__":
    print("[*] Verifying GeometricMapper module...")
    dsp = CSIDSPEngine()
    mapper = GeometricMapper(dsp_engine=dsp, transceiver_origin=(0.0, 0.0))
    
    # Synthesize test peaks representing 4 rectangular room walls at 2.5m distance
    # 2.5m radial distance -> ToF = 2 * 2.5 / c = 16.68 ns
    tof_wall = (2.0 * 2.5 / dsp.c) * 1e9
    test_peaks = [
        (0.0,   tof_wall, 1.0),   # Front wall at angle 0 deg -> (2.5, 0)
        (90.0,  tof_wall, 0.9),   # Top wall at angle 90 deg -> (0, 2.5)
        (180.0, tof_wall, 0.8),   # Back wall at angle 180 deg -> (-2.5, 0)
        (-90.0, tof_wall, 0.85)   # Bottom wall at angle -90 deg -> (0, -2.5)
    ]
    
    coords = mapper.map_peaks_to_cartesian(test_peaks)
    print(f"[+] Mapped 4 wall reflection coordinates:\n{coords[:, :2]}")
    assert len(coords) == 4, "Coordinate transformation failed!"
    assert abs(coords[0, 0] - 2.5) < 0.1, "X-coordinate mapping error!"
    
    # Test 3D Bistatic Ray-Ellipsoid Intersection
    tx_3d = (3.5, 4.0, 2.5)
    rx_3d = (0.0, 0.0, 1.0)
    # Suppose an obstacle is at (1.5, 2.0, 1.5). Let's check distance to TX and RX:
    obs_test = np.array([1.5, 2.0, 1.5])
    d_tx = np.linalg.norm(obs_test - np.array(tx_3d))
    d_rx = np.linalg.norm(obs_test - np.array(rx_3d))
    tof_obs = ((d_tx + d_rx) / dsp.c) * 1e9
    # Arrival direction from RX to obstacle
    vec_rx_obs = obs_test - np.array(rx_3d)
    az_obs = np.degrees(np.arctan2(vec_rx_obs[1], vec_rx_obs[0]))
    el_obs = np.degrees(np.arcsin(vec_rx_obs[2] / d_rx))
    
    bistatic_peaks = [(az_obs, el_obs, tof_obs, 10.0)]
    coords_3d = mapper.map_bistatic_peaks_to_3d(bistatic_peaks, tx_pos_3d=tx_3d, rx_pos_3d=rx_3d)
    print(f"[+] 3D Bistatic mapped obstacle point: {coords_3d[0, :3]}")
    assert np.allclose(coords_3d[0, :3], obs_test, atol=1e-2), "3D bistatic mapping error!"
    
    # Test 3D DBSCAN Obstacle Clustering
    obs_pts_3d = np.array([[1.5 + 0.05*np.random.randn(), 2.0 + 0.05*np.random.randn(), 1.5 + 0.05*np.random.randn(), 1.0] for _ in range(20)])
    clusters_3d = mapper.cluster_3d_obstacles(obs_pts_3d, eps_m=0.5, min_samples=3)
    print(f"[+] Reconstructed 3D obstacle cluster type: {clusters_3d[0]['type']}")
    assert len(clusters_3d) >= 1, "3D DBSCAN clustering failed!"
    print("[+] GeometricMapper verification passed successfully!")

