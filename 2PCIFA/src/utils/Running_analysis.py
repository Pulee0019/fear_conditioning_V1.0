import numpy as np
from utils.logger import log_message
from itertools import groupby


def running_bout_analysis_classify(running_speed,
                                   general_threshold=0.5, 
                                   general_min_duration=0.5, 
                                   rest_min_duration=4, 
                                   pre_locomotion_buffer=5, 
                                   post_locomotion_buffer=5, 
                                   locomotion_duration=2):
    """Analyze running bouts based on locomotion criteria"""
    if running_speed['filtered_speed'] is None:
        speed = running_speed['original_speed']
        log_message("No filtered speed data found. Using original speed data for bout analysis.", "WARNING")
    else:
        speed = running_speed['filtered_speed']
        log_message("Using filtered speed data for bout analysis.")
    
    log_message(f"General running threshold set at {general_threshold:.2f} cm/s")
    
    timestamps = running_speed['timestamps']
    sample_interval = np.mean(np.diff(timestamps))
    sample_rate = int(1.0 / sample_interval)
    threshold = general_threshold
    motion_flag = abs(speed) >= threshold
    groups = groupby(enumerate(motion_flag), key=lambda x: x[1])
    general_bouts = []
    for k, g in groups:
        g = list(g)
        if k and len(g) >= general_min_duration*sample_rate and max([abs(speed[i]) for i, _ in g]) >= 2 * general_threshold:
            idxs = [i for i, _ in g]
            bout_info = [idxs[0], idxs[-1]+2]
            general_bouts.append(bout_info)
    
    combined_bouts = combine_bouts(general_bouts, speed, general_threshold)
    extended_bouts = extend_bouts(combined_bouts, speed, general_threshold)
    final_bouts = combine_bouts(extended_bouts, speed, general_threshold)
    final_bouts_without_edges = exclude_bout_edges(final_bouts, len(speed))
    
    rest_flag = abs(speed) < threshold
    groups = groupby(enumerate(rest_flag), key=lambda x: x[1])
    rest = []
    for k, g in groups:
        g = list(g)
        if k and len(g) >= rest_min_duration*sample_rate:
            idxs = [i for i, _ in g]
            bout_info = [idxs[0], idxs[-1]+2]
            rest.append(bout_info)
            
    locomotion, reset, jerk, other = running_bout_classify(final_bouts_without_edges, speed, timestamps, general_threshold, pre_locomotion_buffer, post_locomotion_buffer, locomotion_duration, sample_rate)
    locomotion = exclude_bout_edges(locomotion, len(speed))
    reset = exclude_bout_edges(reset, len(speed))
    jerk = exclude_bout_edges(jerk, len(speed))
    other = exclude_bout_edges(other, len(speed))
    rest = exclude_bout_edges(rest, len(speed))
    
    bouts = {
        'general_bouts': final_bouts_without_edges,
        'locomotion_bouts': locomotion,
        'reset_bouts': reset,
        'jerk_bouts': jerk,
        'other_bouts': other,
        'rest_bouts': rest
    }

    log_message(f"Identified {len(final_bouts_without_edges)} general bouts. Locomotion: {len(locomotion)}, Reset: {len(reset)}, Jerk: {len(jerk)}, Other: {len(other)}, Rest: {len(rest)}")
    
    return bouts


def running_bout_classify(general_bouts, speed, timestamps, threshold, pre_buffer, post_buffer, duration_buffer, sample_rate):
    '''Classify running bouts into locomotion, reset, jerk, and other'''
    locomotion = []
    reset = []
    jerk = []
    other = []
    duration_buffer_samples = int(duration_buffer * sample_rate)
    for bout in general_bouts:
        start_idx, end_idx = bout
        pre_start = max(0, start_idx - int(pre_buffer * sample_rate))
        post_end = min(len(speed) - 1, end_idx + int(post_buffer * sample_rate))
        duration = end_idx - start_idx + 1
        pre_speed = speed[pre_start:start_idx]
        post_speed = speed[end_idx+1:post_end+1]
        pre_bout_mean = np.mean(abs(pre_speed)) if len(pre_speed) > 0 else 0
        post_bout_mean = np.mean(abs(post_speed)) if len(post_speed) > 0 else 0
        if pre_bout_mean < threshold and post_bout_mean < threshold and duration < duration_buffer_samples:
            jerk.append([start_idx, end_idx])
        elif pre_bout_mean < threshold and post_bout_mean < threshold and duration >= duration_buffer_samples:
            locomotion.append([start_idx, end_idx])
        elif pre_bout_mean >= threshold:
            reset.append([start_idx, end_idx])
        else:
            other.append([start_idx, end_idx])
    return locomotion, reset, jerk, other


def apply_running_filters(speed_data, timestamps, filter_type, **kwargs):
    """
    Apply various filters to running speed data
    
    Parameters:
    - speed_data: array of running speed values
    - timestamps: array of corresponding timestamps
    - filter_type: type of filter to apply ('moving_average', 'median', 'savitzky_golay', 'butterworth')
    - **kwargs: filter-specific parameters
    
    Returns:
    - filtered_speed: filtered speed data
    """
    import numpy as np
    from scipy.signal import savgol_filter, butter, filtfilt
    
    if filter_type == 'moving_average':
        window_size = kwargs.get('window_size', 5)
        if window_size % 2 == 0:
            window_size += 1  # Make window size odd
        
        # Apply moving average filter
        kernel = np.ones(window_size) / window_size
        filtered_speed = np.convolve(speed_data, kernel, mode='same')
        
    elif filter_type == 'median':
        window_size = kwargs.get('window_size', 5)
        if window_size % 2 == 0:
            window_size += 1  # Make window size odd
        
        # Apply median filter
        filtered_speed = []
        half_window = window_size // 2
        
        for i in range(len(speed_data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(speed_data), i + half_window + 1)
            window_data = speed_data[start_idx:end_idx]
            filtered_speed.append(np.median(window_data))
        
        filtered_speed = np.array(filtered_speed)
        
    elif filter_type == 'savitzky_golay':
        window_size = kwargs.get('window_size', 11)
        poly_order = kwargs.get('poly_order', 3)
        
        if window_size % 2 == 0:
            window_size += 1  # Make window size odd
        
        # Apply Savitzky-Golay filter
        filtered_speed = savgol_filter(speed_data, window_size, poly_order)
        
    elif filter_type == 'butterworth':
        sampling_rate = kwargs.get('sampling_rate', 10)  # Hz
        cutoff_freq = kwargs.get('cutoff_freq', 2.0)  # Hz
        filter_order = kwargs.get('filter_order', 2)
        
        # Design Butterworth low-pass filter
        nyquist_freq = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
        
        # Apply zero-phase filtering
        filtered_speed = filtfilt(b, a, speed_data)
        
    else:
        # No filtering
        filtered_speed = speed_data.copy()
    
    return filtered_speed


def preprocess_running_data(ast2_data, filter_settings):
    """
    Preprocess running data with specified filters
    
    Parameters:
    - ast2_data: dictionary containing running data
    - filter_settings: dictionary with filter configuration
    
    Returns:
    - processed_data: dictionary with original and filtered data
    """
    if ast2_data is None:
        return None
    
    try:
        speed = ast2_data['data']['speed']
        timestamps = ast2_data['data']['timestamps']
        
        # Apply filters sequentially
        filtered_speed = speed.copy()
        filter_history = ["original"]
        
        for filter_config in filter_settings:
            filter_type = filter_config['type']
            if filter_type != 'none':
                filtered_speed = apply_running_filters(
                    filtered_speed, timestamps, 
                    filter_type, **filter_config.get('params', {})
                )
                filter_history.append(filter_type)
        
        # Create processed data structure
        processed_data = {
            'original_speed': speed,
            'filtered_speed': filtered_speed,
            'timestamps': timestamps,
            'filter_history': filter_history,
            'filter_settings': filter_settings
        }
        
        return processed_data
        
    except Exception as e:
        log_message(f"Error in running data preprocessing: {str(e)}", "ERROR")
        return None


def exclude_bout_edges(bouts, data_length):
    """Exclude bouts that start at the first data point or end at the last data point"""
    filtered_bouts = []
    for bout in bouts:
        start_idx, end_idx = bout
        if start_idx > 0 and end_idx < data_length - 1:
            filtered_bouts.append(bout)
    return filtered_bouts


def combine_bouts(bouts, speed, general_threshold, max_gap=15, max_exceed_ratio=0.2):
    """Combine bouts that are close in time or have significant speed signals in the gap between them"""
    combined_bouts = []
    i = 0
    while i < len(bouts):
        bout = bouts[i]
        while i + 1 < len(bouts):
            next_bout = bouts[i + 1]
            gap_start = bout[1]
            gap_end = next_bout[0]
            gap = gap_end - gap_start

            time_condition = gap <= max_gap
            
            signal_condition = False
            if gap > 0:
                gap_speed = speed[gap_start:gap_end]
                high_ratio = np.sum(gap_speed >= general_threshold) / len(gap_speed)
                low_ratio = np.sum(gap_speed <= -general_threshold) / len(gap_speed)
                signal_condition = high_ratio  + low_ratio > max_exceed_ratio

            if time_condition or signal_condition:
                bout = [bout[0], next_bout[1]]
                i += 1
            else:
                break
        combined_bouts.append(bout)
        i += 1
    return combined_bouts


def extend_bouts(combined_bouts, signal, general_threshold):
    """Extend bouts to include adjacent points that are part of the same trend and above threshold"""
    extended_bouts = []

    for bout in combined_bouts:
        onset, offset = bout[0], bout[1]

        # Extend onset backward
        new_onset = onset
        if onset > 0:
            # Find trend turning point backward
            trend_start = onset
            i = onset

            if signal[i] > signal[i-1]:
                curr_trend = 'increasing'
            elif signal[i] < signal[i-1]:
                curr_trend = 'decreasing'
            else:
                curr_trend = None

            while i > 0:
                prev_val = signal[i-1]
                curr_val = signal[i]

                if curr_val > prev_val:
                    step_trend = 'increasing'
                elif curr_val < prev_val:
                    step_trend = 'decreasing'
                else:
                    step_trend = curr_trend

                if step_trend != curr_trend or step_trend is None:
                    break

                trend_start = i - 1
                i -= 1

            # If still above threshold, continue backward to find point just below threshold
            if signal[trend_start] >= general_threshold or signal[trend_start] <= -general_threshold:
                new_onset = trend_start
                i = trend_start

                while i > 0:
                    prev_val = signal[i-1]
                    curr_val = signal[i]

                    # If current value is above threshold, update new_onset
                    if abs(curr_val) >= general_threshold:
                        new_onset = i

                    # If previous value is below threshold, stop
                    if abs(prev_val) < general_threshold:
                        break

                    i -= 1

        # Extend offset forward
        new_offset = offset
        if offset < len(signal) - 1:
            # Find trend turning point forward
            trend_end = offset
            i = offset

            if signal[i+1] > signal[i]:
                curr_trend = 'increasing'
            elif signal[i+1] < signal[i]:
                curr_trend = 'decreasing'
            else:
                curr_trend = None

            while i < len(signal) - 1:
                curr_val = signal[i]
                next_val = signal[i+1]

                if next_val > curr_val:
                    step_trend = 'increasing'
                elif next_val < curr_val:
                    step_trend = 'decreasing'
                else:
                    step_trend = curr_trend

                if step_trend != curr_trend or step_trend is None:
                    break

                trend_end = i + 1
                i += 1

            # If still above threshold, continue forward to find point just below threshold
            if signal[trend_end] >= general_threshold or signal[trend_end] <= -general_threshold:
                new_offset = trend_end
                i = trend_end

                while i < len(signal) - 1:
                    curr_val = signal[i]
                    next_val = signal[i+1]

                    # If current value is above threshold, update new_offset
                    if abs(curr_val) >= general_threshold:
                        new_offset = i

                    # If next value is below threshold, stop
                    if abs(next_val) < general_threshold:
                        break

                    i += 1

        extended_bouts.append([new_onset, new_offset])

    return extended_bouts