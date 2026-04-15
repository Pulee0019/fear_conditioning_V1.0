import os
import re
import numpy as np
from scipy.stats import scoreatpercentile

SESSION_ORDER = {
    'Hab': 0,
    'FC': 1,
    'EXT1': 2,
    'EXT2': 3,
    'EXT3': 4
}

def extract_session_key(folder):
    """
    Extracts the session type from the folder name and returns its corresponding order based on SESSION_ORDER.
    """
    parts = folder.split('_')
    if len(parts) >= 2:
        session_type = parts[1]
        return SESSION_ORDER.get(session_type, 999)
    return 999

def extract_number(folder_name):
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else 0

def calculate_t2p_dff(path, frame_rates=None):
    folders = os.listdir(path)
    folders_sorted = sorted(folders, key=extract_number)
    paired_paths = [os.path.join(path, folder, 'track2p', 'matched_suite2p') for folder in folders_sorted]
    s2p_paired_folders = [os.listdir(p) for p in paired_paths]
    s2p_paired_folders_sorted = [sorted(folders, key=extract_session_key) for folders in s2p_paired_folders]
    s2p_paired_paths = [[os.path.join(paired_paths[i], folder) for folder in s2p_paired_folders_sorted[i]] for i in range(len(s2p_paired_folders_sorted))]
    s2p_paired_fs = [[np.load(os.path.join(s2p_paired_paths[i][j], 'suite2p', 'plane0', 'F.npy'), allow_pickle=True) for j in range(len(s2p_paired_paths[i]))] for i in range(len(s2p_paired_paths))]
    s2p_paired_fneus = [[np.load(os.path.join(s2p_paired_paths[i][j], 'suite2p', 'plane0', 'Fneu.npy'), allow_pickle=True) for j in range(len(s2p_paired_paths[i]))] for i in range(len(s2p_paired_paths))]
    paired_intensity = [[s2p_paired_fs[i][j] - 0.7 * s2p_paired_fneus[i][j] for j in range(len(s2p_paired_fs[i]))] for i in range(len(s2p_paired_fs))]
    paired_dff = []
    for i in range(len(paired_intensity)):
        folder_dff = []
        for j in range(len(paired_intensity[i])):
            frame_rate = frame_rates[i][j] if frame_rates is not None else 3.48772321428571
            intensity = paired_intensity[i][j]
            dff = np.zeros_like(intensity)
            for k in range(intensity.shape[0]):
                intensity_cell = intensity[k]
                [baseline_window, _, _, _] = std_based_baseline_windows(intensity_cell, window_size=int(frame_rate * 6))
                baseline = intensity_cell[baseline_window[0][0]:baseline_window[0][1]]
                dff_cell = (intensity_cell - baseline.mean()) / baseline.mean()
                dff[k] = dff_cell
            folder_dff.append(dff)
        paired_dff.append(folder_dff)

    return paired_dff, paired_intensity

def std_based_baseline_windows(data, window_size, step_size=None):
    n = len(data)

    if step_size is None:
        step_size = int(window_size / 4)
    
    if window_size > n:
        window_size = n
    
    if step_size < 1:
        step_size = 1
    
    num_windows = int(np.floor((n - window_size) / step_size)) + 1
    
    if num_windows < 1:
        return [], np.zeros(n, dtype=bool), np.array([]), None
    
    std_values = np.zeros(num_windows)
    window_centers = np.zeros(num_windows)
    
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size - 1, n - 1)
        
        window_data = data[start_idx:end_idx + 1]
        std_values[i] = np.std(window_data)
        window_centers[i] = (start_idx + end_idx) / 2
    
    valid_std = std_values[~np.isnan(std_values) & ~np.isinf(std_values)]
    if len(valid_std) == 0:
        q10_std = np.nan
    else:
        q10_std = scoreatpercentile(valid_std, 10)
    
    baseline_window_indices = np.where(std_values <= q10_std)[0]
    
    baseline_mask = np.zeros(n, dtype=bool)
    for idx in baseline_window_indices:
        start_idx = idx * step_size
        end_idx = min(start_idx + window_size - 1, n - 1)
        
        if np.isnan(start_idx) or np.isnan(end_idx) or start_idx < 0 or end_idx < 0:
            continue
        
        start_idx = int(max(0, start_idx))
        end_idx = int(min(n - 1, end_idx))
        
        if start_idx <= end_idx and start_idx >= 0 and end_idx < n:
            baseline_mask[start_idx:end_idx + 1] = True
    
    baseline_windows = []
    for idx in baseline_window_indices:
        start_idx = idx * step_size
        end_idx = min(start_idx + window_size - 1, n - 1)
        
        if np.isnan(start_idx) or np.isnan(end_idx) or start_idx < 0 or end_idx < 0:
            continue
        
        start_idx = int(max(0, start_idx))
        end_idx = int(min(n - 1, end_idx))
        
        if start_idx <= end_idx and start_idx >= 0 and end_idx < n:
            baseline_windows.append([start_idx, end_idx])
    
    return baseline_windows, baseline_mask, std_values, q10_std