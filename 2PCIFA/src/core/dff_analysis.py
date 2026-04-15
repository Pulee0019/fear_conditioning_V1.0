import matplotlib.pyplot as plt
import numpy as np
from utils.Running_analysis import *
from utils.statistic import *
from scipy.interpolate import interp1d
from core.calculate_overall_dff import std_based_baseline_windows

SESSION_INDEX = {
    0: 'Hab',
    1: 'EXT1'
}

def plot_overall_dff_and_running(dffs, runnings, imaging_frame_rates=[], running_sample_rates=[], events=[], plot_pre=5, plot_post=5, **kwargs):
    filter_settings = [{'type': 'moving_average', 'params': {'window_size': kwargs.get("smooth_window", 10)}}]
    for i, paired in enumerate(dffs):
        for j, session_dff in enumerate(paired):
            imaging_frame_rate = imaging_frame_rates[i][j]
            running_sample_rate = running_sample_rates[i][j]
            event_data = events[i][j]
            plt.figure(figsize=(16, 1))
            running_data, aligned_time = runnings[i][j]
            running_with_time = {'speed': running_data, 'timestamps': aligned_time}
            para_running_data = {'data': running_with_time}
            processed_running_data = preprocess_running_data(para_running_data, filter_settings)
            smoothed_running = processed_running_data['filtered_speed']
            bouts = running_bout_analysis_classify(processed_running_data,
                                                                        general_threshold=kwargs.get("general_threshold", 0.5),
                                                                        general_min_duration=kwargs.get("general_min_duration", 0.5),
                                                                        rest_min_duration=kwargs.get("rest_min_duration", 4),
                                                                        pre_locomotion_buffer=kwargs.get("pre_locomotion_buffer", 5),
                                                                        post_locomotion_buffer=kwargs.get("post_locomotion_buffer", 5),
                                                                        locomotion_duration=kwargs.get("locomotion_duration", 2))
            
            for _, row in event_data.iterrows():
                if row['Event Type'] == 0:
                    color = "#ffffff"
                elif row['Event Type'] == 3:
                    color = "#ffffff"
                else:
                    color = "#000000"
                start_col = 'start_time'
                end_col = 'end_time'
                start_time = row[start_col]
                end_time = row[end_col]
                plt.axvspan(start_time*running_sample_rate, end_time*running_sample_rate, color=color, alpha=0.5)
            
            plt.plot(smoothed_running, color='#000000')
            for start, end in bouts['general_bouts']:
                plt.axvspan(start, end, color="#FF0000FF", alpha=0.3)
                
            plt.title(f'Group {i+1} Session {j+1} Running Data')
            plt.xlabel('Time (frames)')
            plt.xlim(0, len(running_data))
            plt.ylabel('Speed (cm/s)')
            plt.show()

            plt.figure(figsize=(16, 9))
            for _, row in event_data.iterrows():
                if row['Event Type'] == 0:
                    color = "#ffffff"
                elif row['Event Type'] == 3:
                    color = "#ffffff"
                else:
                    color = "#000000"
                start_time = row['start_time']
                end_time = row['end_time']
                plt.axvspan(start_time*imaging_frame_rate, end_time*imaging_frame_rate, color=color, alpha=0.5)
                
            shift = 5
            for n in range(session_dff.shape[0]):
                dff = session_dff[n, :]
                plt.plot(dff + shift * n, color='k')
                
                    
                [baseline_window, _, _, _] = std_based_baseline_windows(dff, window_size=21)
                baseline = dff[baseline_window[0][0]:baseline_window[0][1]]
                baseline_mean = np.mean(baseline)
                baseline_std = np.std(baseline)
                threshold = baseline_mean + 3 * baseline_std
                plt.axhline(y=threshold + n * shift, color='k', linestyle='--', alpha=0.3)
                plt.plot(baseline_window[0], [threshold + n * shift, threshold + n * shift], 'r-')
                for m, (start, end) in enumerate(bouts['general_bouts']):
                    if start - plot_pre * running_sample_rate < 0 or start + plot_post * running_sample_rate > len(running_data):
                        continue
                    if end - plot_pre * running_sample_rate < 0 or end + plot_post * running_sample_rate > len(running_data):
                        continue
                    plt.plot(range(int(start*imaging_frame_rate/running_sample_rate), int(end*imaging_frame_rate/running_sample_rate)),
                            dff[int(start*imaging_frame_rate/running_sample_rate):int(end*imaging_frame_rate/running_sample_rate)] + shift * n, color='red', alpha=0.7)
                    
            plt.title(f'Group {i+1} Session {j+1} ΔF/F {session_dff.shape[0]} cells')
            plt.xlabel('Time (frames)')
            plt.xlim(0, session_dff.shape[1])
            plt.ylabel('ΔF/F')
            plt.show()
            
            fig = plt.figure(figsize=(16, 4))
            ax_main = fig.add_axes([0.1, 0.3, 0.8, 0.6])  
            ax_cbar = fig.add_axes([0.1, 0.1, 0.8, 0.05])  

            plt.sca(ax_main)
            for m, (start, end) in enumerate(bouts['general_bouts']):
                e_running_start = start * imaging_frame_rate / running_sample_rate
                e_running_end = end * imaging_frame_rate / running_sample_rate
                plt.axvspan(e_running_start, e_running_end, color="#FF0000FF", alpha=0.3)
            heatmap = plt.imshow(session_dff, aspect='auto', cmap='hot', interpolation='nearest',
                                origin='lower', extent=[0, session_dff.shape[1], 0, session_dff.shape[0]])
            plt.clim(-5, 30)
            plt.title(f'Paired {i+1}, Session {j+1}, {session_dff.shape[0]} Cells ΔF/F Heatmap')
            plt.xlabel('Time (frames)')
            plt.ylabel('Cells')
            plt.colorbar(heatmap, cax=ax_cbar, orientation='horizontal', label='ΔF/F')
            plt.axis('off')
            plt.show()


def calculate_event_dff_from_intensity(intensitys, runnings, imaging_frame_rates=[], running_sample_rates=[], events=[], plot_pre=5, plot_post=5, **kwargs):
    filter_settings = [{'type': 'moving_average', 'params': {'window_size': kwargs.get("smooth_window", 10)}}]
    intensitys = [list(row) for row in zip(*intensitys)]
    runnings = [list(row) for row in zip(*runnings)]
    imaging_frame_rates = [list(row) for row in zip(*imaging_frame_rates)]
    running_sample_rates = [list(row) for row in zip(*running_sample_rates)]
    events = [list(row) for row in zip(*events)]
    exp_dffs = []
    exp_runnings = []
    for i, paired in enumerate(intensitys):
        events_dffs = []
        events_runnings = []
        for j, session_intensity in enumerate(paired):
            imaging_frame_rate = imaging_frame_rates[i][j]
            running_sample_rate = running_sample_rates[i][j]
            running_data, aligned_time = runnings[i][j]
            event_data = events[i][j]
            running_with_time = {'speed': running_data, 'timestamps': aligned_time}
            para_running_data = {'data': running_with_time}
            processed_running_data = preprocess_running_data(para_running_data, filter_settings)
            smoothed_running = processed_running_data['filtered_speed']
            bouts = running_bout_analysis_classify(processed_running_data,
                                                    general_threshold=kwargs.get("general_threshold", 0.5),
                                                    general_min_duration=kwargs.get("general_min_duration", 0.5),
                                                    rest_min_duration=kwargs.get("rest_min_duration", 4),
                                                    pre_locomotion_buffer=kwargs.get("pre_locomotion_buffer", 5),
                                                    post_locomotion_buffer=kwargs.get("post_locomotion_buffer", 5),
                                                    locomotion_duration=kwargs.get("locomotion_duration", 2)
                                                    )
            
            session_runnings = []
            session_event_dffs = []
            target_frame_rate = min(min(imaging_frame_rates))
            for _, row in event_data.iterrows():
                if row['Event Type'] == 1:
                    sound_start = row['start_time']
                    sound_end = row['end_time']
                    is_running = False
                    for start, end in bouts['general_bouts']:
                        if (start >= (sound_start-plot_pre)*running_sample_rate and start <= sound_end*running_sample_rate) or (end >= (sound_start-plot_pre)*running_sample_rate and end <= sound_end*running_sample_rate):
                            is_running = True
                            break
                    if is_running:
                        print(f"Skipping event at {sound_start}s to {sound_end}s due to running")
                        continue
                    sound_running = smoothed_running[int(sound_start*running_sample_rate)-int(plot_pre*running_sample_rate):int(sound_end*running_sample_rate)+int(plot_post*running_sample_rate)]
                    session_runnings.append(sound_running)
                    cell_event_dffs = []
                    for n in range(session_intensity.shape[0]):
                        # if n != 2:
                        #     continue
                        cell_event_intensity = session_intensity[n, int(sound_start*target_frame_rate)-int(plot_pre*target_frame_rate):int(sound_end*target_frame_rate)+int(plot_post*target_frame_rate)]
                        session_baseline_intensity = session_intensity[n, int(sound_start*target_frame_rate)-int(plot_pre*target_frame_rate):int(sound_start*target_frame_rate)]
                        baseline_mean = np.mean(session_baseline_intensity)
                        cell_event_dff = (cell_event_intensity - baseline_mean) / baseline_mean
                        cell_event_dffs.append(cell_event_dff)
                        
                    session_event_dffs.append(cell_event_dffs)
            
            events_runnings.append(session_runnings)
            events_dffs.append(session_event_dffs)
            
        exp_dffs.append(events_dffs)
        exp_runnings.append(events_runnings)

    return exp_dffs, exp_runnings


def calculate_event_dff_from_dff(dffs, runnings, imaging_frame_rates=[], running_sample_rates=[], events=[], time_window=1, pre_duration=2, post_duration=2, plot_pre=5, plot_post=5, **kwargs):
    filter_settings = [{'type': 'moving_average', 'params': {'window_size': kwargs.get("smooth_window", 10)}}]
    dffs = [list(row) for row in zip(*dffs)]
    runnings = [list(row) for row in zip(*runnings)]
    imaging_frame_rates = [list(row) for row in zip(*imaging_frame_rates)]
    running_sample_rates = [list(row) for row in zip(*running_sample_rates)]
    events = [list(row) for row in zip(*events)]
    exp_dffs = []
    exp_runnings = []
    exp_ensemble_sizes = []
    for i, paired in enumerate(dffs):
        events_dffs = []
        events_runnings = []
        events_ensemble_sizes = []
        for j, session_dff in enumerate(paired):
            imaging_frame_rate = imaging_frame_rates[i][j]
            running_sample_rate = running_sample_rates[i][j]
            running_data, aligned_time = runnings[i][j]
            event_data = events[i][j]
            running_with_time = {'speed': running_data, 'timestamps': aligned_time}
            para_running_data = {'data': running_with_time}
            processed_running_data = preprocess_running_data(para_running_data, filter_settings)
            smoothed_running = processed_running_data['filtered_speed']
            bouts = running_bout_analysis_classify(processed_running_data,
                                                    general_threshold=kwargs.get("general_threshold", 0.5),
                                                    general_min_duration=kwargs.get("general_min_duration", 0.5),
                                                    rest_min_duration=kwargs.get("rest_min_duration", 4),
                                                    pre_locomotion_buffer=kwargs.get("pre_locomotion_buffer", 5),
                                                    post_locomotion_buffer=kwargs.get("post_locomotion_buffer", 5),
                                                    locomotion_duration=kwargs.get("locomotion_duration", 2)
                                                    )

            session_runnings = []
            session_dffs = []
            active_cell = 0
            target_frame_rate = min(min(imaging_frame_rates))
            for _, row in event_data.iterrows():
                if row['Event Type'] == 1:
                    sound_start = row['start_time']
                    sound_end = row['end_time']
                    is_running = False
                    for start, end in bouts['general_bouts']:
                        if (start >= (sound_start-plot_pre)*running_sample_rate and start <= sound_end*running_sample_rate) or (end >= (sound_start-plot_pre)*running_sample_rate and end <= sound_end*running_sample_rate):
                            is_running = True
                            break
                    if is_running:
                        print(f"Skipping event at {sound_start}s to {sound_end}s due to running")
                        continue
                    sound_running = smoothed_running[int(sound_start*running_sample_rate)-int(plot_pre*running_sample_rate):int(sound_end*running_sample_rate)+int(plot_post*running_sample_rate)]
                    session_runnings.append(sound_running)
                    
                    cell_dffs = []
                    for n in range(session_dff.shape[0]):
                        dff = session_dff[n, :]
                        x = np.arange(len(dff))
                        f = interp1d(x, dff, kind='linear')
                        x_new = np.linspace(0, len(dff)-1, num=int(len(dff)*target_frame_rate/imaging_frame_rate))
                        dff = f(x_new)
                        [baseline_window, _, _, _] = std_based_baseline_windows(dff, window_size=21)
                        baseline = dff[baseline_window[0][0]:baseline_window[0][1]]
                        baseline_mean = np.mean(baseline)
                            
                        baseline_std = np.std(baseline)
                        threshold = baseline_mean + 3 * baseline_std
                        
                        sound_dff = dff[int(sound_start*target_frame_rate)-int(plot_pre*target_frame_rate):int(sound_end*target_frame_rate)+int(plot_post*target_frame_rate)]
                        cell_dffs.append(sound_dff)
                        
                        sound_start_dff = dff[int(sound_start*target_frame_rate)-int(pre_duration*target_frame_rate):int(sound_start*target_frame_rate)+int(post_duration*target_frame_rate)]
                        
                        sound_start_groups = groupby(enumerate(sound_start_dff > threshold), key=lambda x: x[1])
                        sound_start_active_idx = []
                        for k, g in sound_start_groups:
                            g = list(g)
                            if k and len(g) >= int(time_window*target_frame_rate):
                                idxs = [i for i, _ in g]
                                if sound_start_dff[idxs].max() > baseline_mean + 6 * baseline_std:
                                    sound_start_active_idx.extend(idxs)
                                    
                        if sound_start_active_idx:
                            active_cell += 1
                            
                    session_dffs.append(cell_dffs)
            
            events_runnings.append(session_runnings)
            events_dffs.append(session_dffs)
            events_ensemble_sizes.append(active_cell / session_dff.shape[0] / len(event_data[event_data['Event Type'] == 1]))
        exp_dffs.append(events_dffs)
        exp_runnings.append(events_runnings)
        exp_ensemble_sizes.append(events_ensemble_sizes)
        
    return exp_dffs, exp_runnings, exp_ensemble_sizes


def find_first_valid_shape(data):
    if data is None:
        return None
    
    if isinstance(data, np.ndarray) and data.shape[0] > 0:
        return data.shape[0]
    
    if isinstance(data, (list, tuple)):
        for item in data:
            result = find_first_valid_shape(item)
            if result is not None:
                return result
    
    return None


def event_dff_analysis(data, plot_pre=5, plot_post=5, imaging_frame_rates=[], cell_type="", fig_save_path="", fig_save_format="png"):
    imaging_frame_rate = min(min(imaging_frame_rates))
    dff_length = find_first_valid_shape(data)
    print(dff_length)
    if dff_length is None:
        raise ValueError("No valid dff data found")

    for i, paired in enumerate(data):
        if any(session_dffs is None for session_dffs in paired):
            continue
        pre_mean_dffs = []
        post_mean_dffs = []
        pre_peak_dffs = []
        post_peak_dffs = []
        pre_auc_dffs = []
        post_auc_dffs = []
        for j, session_dff in enumerate(paired):
            for k, cell_dffs in enumerate(session_dff):
                for n, dff in enumerate(cell_dffs):
                    original_time_stamps = np.arange(len(dff))
                    f = interp1d(original_time_stamps, dff, kind='linear', 
                                fill_value='extrapolate', bounds_error=False)
                    x_new = np.arange(dff_length)
                    dff = f(x_new)
                    pre_dff = dff[:int(plot_pre*imaging_frame_rate)]
                    post_dff = dff[int(plot_pre*imaging_frame_rate):int((2*plot_pre)*imaging_frame_rate)]
                    pre_mean_dffs.append(np.mean(pre_dff))
                    post_mean_dffs.append(np.mean(post_dff))
                    pre_peak_dffs.append(np.max(pre_dff))
                    post_peak_dffs.append(np.max(post_dff))
                    pre_auc_dffs.append(np.trapz(pre_dff, dx=1/imaging_frame_rate))
                    post_auc_dffs.append(np.trapz(post_dff, dx=1/imaging_frame_rate))
                    
        mean_dffs = [pre_mean_dffs, post_mean_dffs]
        peak_dffs = [pre_peak_dffs, post_peak_dffs]
        auc_dffs = [pre_auc_dffs, post_auc_dffs]
        plot_paired_statistics(list(zip(*mean_dffs)), "#09af25", ylabel="Average %DF/F", type=cell_type, data_type="dff", save_path=fig_save_path, ylim=None, xticks=["Baseline", "Sound"], figure_save_format=fig_save_format)
        plot_paired_statistics(list(zip(*peak_dffs)), "#09af25", ylabel="Peak %DF/F", type=cell_type, data_type="dff_peak", save_path=fig_save_path, ylim=None, xticks=["Baseline", "Sound"], figure_save_format=fig_save_format)
        plot_paired_statistics(list(zip(*auc_dffs)), "#09af25", ylabel="AUC %DF/F", type=cell_type, data_type="dff_auc", save_path=fig_save_path, ylim=None, xticks=["Baseline", "Sound"], figure_save_format=fig_save_format)


def event_running_analysis(data, plot_pre=5, plot_post=5, running_sample_rates=[], cell_type="", fig_save_path="", fig_save_format="png"):
    running_frame_rate = min(min(running_sample_rates))
    speed_length = find_first_valid_shape(data)
    if speed_length is None:
        raise ValueError("No valid running data found")
    
    for i, paired in enumerate(data):
        pre_mean_runnings = []
        post_mean_runnings = []
        if any(session_runnings is None for session_runnings in paired):
            continue
        for j, session_runnings in enumerate(paired):
            event_runings = []
            for k, running in enumerate(session_runnings):
                original_time_stamps = np.arange(len(running))
                f = interp1d(original_time_stamps, running, kind='linear', 
                            fill_value='extrapolate', bounds_error=False)
                x_new = np.arange(speed_length)
                running = f(x_new)
                pre_running = running[:int(plot_pre*running_frame_rate)]
                post_running = running[int(plot_pre*running_frame_rate):int((2*plot_pre)*running_frame_rate)]
                pre_mean_runnings.append(np.mean(pre_running))
                post_mean_runnings.append(np.mean(post_running))

        mean_runnings = [pre_mean_runnings, post_mean_runnings]
        plot_paired_statistics(list(zip(*mean_runnings)), "#000000", ylabel="Average Speed (cm/s)", type=cell_type, data_type="speed", save_path=fig_save_path, ylim=None, xticks=["Baseline", "Sound"], figure_save_format=fig_save_format)


def plot_event_dff(dffs_trace, imaging_frame_rates, plot_pre, group_name, color, y_limit, save_path, figure_save_format):
    imaging_frame_rate = min(min(imaging_frame_rates))
    dff_length = find_first_valid_shape(dffs_trace)
    
    if dff_length is None:
        raise ValueError("No valid dff data found")
    dff_time_stamps = np.linspace(-plot_pre, -plot_pre + (dff_length-1)/imaging_frame_rate, dff_length)
    
    for i, paired in enumerate(dffs_trace):
        if any(session_dffs is None for session_dffs in paired):
            continue
        exp_dffs = []
        for j, session_dff in enumerate(paired):
            for k, cell_dffs in enumerate(session_dff):
                for n, dff in enumerate(cell_dffs):
                    original_time_stamps = np.arange(len(dff))
                    f = interp1d(original_time_stamps, dff, kind='linear', 
                                fill_value='extrapolate', bounds_error=False)
                    x_new = np.arange(dff_length)
                    exp_dffs.append(f(x_new))
                        
                        
        mean_dff = np.mean(exp_dffs, axis=0)
        sem_dff = np.std(exp_dffs, axis=0) / np.sqrt(len(exp_dffs))
        plt.figure(figsize=(6, 3))
        plt.plot(dff_time_stamps, mean_dff, color=color)
        plt.axhline(0, color='k', linestyle='--')
        plt.axvline(0, color='k', linestyle='--')
        plt.fill_between(dff_time_stamps, mean_dff - sem_dff, mean_dff + sem_dff, color=color, alpha=0.3)
        plt.title(f'Mean ΔF/F - {group_name} - {SESSION_INDEX[i]} - {len(cell_dffs)} cells')
        plt.xlabel('Time (s)')
        plt.ylabel('ΔF/F')
        if y_limit is not None:
            plt.ylim(y_limit)
        plt.tight_layout()
        plt.savefig(rf'{save_path}\mean_trace_dff_' + group_name + '_' + SESSION_INDEX[i] + f".{figure_save_format}", dpi=300)
        plt.show()


def plot_event_running(runnings_trace, running_sample_rates, plot_pre, group_name, color, y_limit, save_path, figure_save_format):
    running_frame_rate = min(min(running_sample_rates))
    running_length = find_first_valid_shape(runnings_trace)
    if running_length is None:
        raise ValueError("No valid running data found")
    running_time_stamps = np.linspace(-plot_pre, -plot_pre + (running_length-1)/running_frame_rate, running_length)
        
    for i, paired in enumerate(runnings_trace):
        if any(session_runnings is None for session_runnings in paired):
            continue
        exp_runnings = []
        for j, session_runnings in enumerate(paired):
            for k, running in enumerate(session_runnings):
                original_time_stamps = np.arange(len(running))
                f = interp1d(original_time_stamps, running, kind='linear', 
                            fill_value='extrapolate', bounds_error=False)
                x_new = np.arange(running_length)
                exp_runnings.append(f(x_new))
                running_time_stamps = np.linspace(-plot_pre, -plot_pre + (running_length-1)/running_frame_rate, running_length)
                    
        mean_running = np.mean(exp_runnings, axis=0)
        sem_running = np.std(exp_runnings, axis=0) / np.sqrt(len(exp_runnings))
        plt.figure(figsize=(6, 3))
        plt.plot(running_time_stamps, mean_running, color=color)
        plt.axhline(0, color='k', linestyle='--')
        plt.axvline(0, color='k', linestyle='--')
        plt.fill_between(running_time_stamps, mean_running - sem_running, mean_running + sem_running, color=color, alpha=0.3)
        plt.title(f'Mean Running Speed - {group_name} - {SESSION_INDEX[i]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (cm/s)')
        if y_limit is not None:
            plt.ylim(y_limit)
        plt.tight_layout()
        plt.savefig(rf'{save_path}\mean_trace_running_' + group_name + '_' + SESSION_INDEX[i] + f".{figure_save_format}", dpi=300)
        plt.show()