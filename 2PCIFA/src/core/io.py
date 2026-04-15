import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os
import re

from utils.logger import log_message


def convert_num(s):
    s = s.strip()
    try:
        if '.' in s or 'e' in s or 'E' in s:
            return float(s)
        else:
            return int(s)
    except ValueError:
        return s


def h_AST2_readData(filename):
    header = {}
    
    with open(filename, 'rb') as fid:
        header_lines = []
        while True:
            line = fid.readline().decode('utf-8').strip()
            if line == 'header_end':
                break
            header_lines.append(line)
        
        for line in header_lines:
            match = re.match(r"header\.(\w+)\s*=\s*(.*);$", line)
            if not match:
                continue
            key = match.group(1)
            value_str = match.group(2).strip()
            
            if value_str.startswith("'") and value_str.endswith("'"):
                header[key] = value_str[1:-1]
            elif value_str.startswith('[') and value_str.endswith(']'):
                inner = value_str[1:-1].strip()
                if not inner:
                    header[key] = []
                else:
                    if ';' in inner:
                        rows = inner.split(';')
                        array = []
                        for row in rows:
                            row = row.strip()
                            if row:
                                elements = row.split()
                                array.append([convert_num(x) for x in elements])
                        header[key] = array
                    else:
                        elements = inner.split()
                        header[key] = [convert_num(x) for x in elements]
            else:
                header[key] = convert_num(value_str)
        
        binary_data = np.fromfile(fid, dtype=np.int16)
    
    if 'activeChIDs' in header and 'scale' in header:
        if isinstance(header['activeChIDs'], list):
            numOfCh = len(header['activeChIDs'])
        elif isinstance(header['activeChIDs'], int):
            numOfCh = 1
        data = binary_data.reshape((numOfCh, -1), order='F') / header['scale']
    else:
        data = None

    return header, data


def h_AST2_raw2Speed(rawData, info, voltageRange=None, invert_running=True, threadmill_diameter=16):
    if voltageRange is None or len(voltageRange) == 0:
        voltageRange = h_calibrateVoltageRange(rawData)
    
    speedDownSampleFactor = info['saveEvery']
    
    rawDataLength = len(rawData)
    segmentLength = speedDownSampleFactor
    speedDataLength = rawDataLength // segmentLength
    
    if rawDataLength % segmentLength != 0:
        log_message(f"SpeedDataLength is not integer!  speedDataLength = {rawDataLength}, speedDownSampleFactor = {segmentLength}", "ERROR")
        rawData = rawData[:speedDataLength * segmentLength]
    
    t = ((np.arange(speedDataLength) + 1) * speedDownSampleFactor) / info['inputRate']
    time_segment = (np.arange(segmentLength) + 1) / info['inputRate']
    reshapedData = rawData.reshape(segmentLength, speedDataLength, order='F')
    speedData2 = h_computeSpeed(time_segment, reshapedData, voltageRange, threadmill_diameter=threadmill_diameter)
    
    if invert_running:
        speedData2 = -speedData2
    
    speedData = {
        'timestamps': t,
        'speed': speedData2
    }
    
    return speedData


def h_calibrateVoltageRange(rawData):
    peakValue, peakPos = h_AST2_findPeaks(rawData)
    valleyValue, valleyPos = h_AST2_findPeaks(-rawData)
    valleyValue = [-x for x in valleyValue]
    
    if len(peakValue) > 0 and len(valleyValue) > 0:
        voltageRange = [np.mean(valleyValue), np.mean(peakValue)]
        if np.diff(voltageRange) > 3:
            log_message(f"Calibrated voltage range is {voltageRange}")
        else:
            log_message("Calibration error. Range too small")
            voltageRange = [0, 5]
    else:
        voltageRange = [0, 5]
        log_message("Calibration fail! Return default: [0 5].")
    
    return voltageRange


def h_AST2_findPeaks(data):
    transitionPos = np.where(np.abs(np.diff(data)) > 2)[0]
    
    transitionPos = transitionPos[(transitionPos > 50) & (transitionPos < len(data) - 50)]
    
    if len(transitionPos) >= 1:
        peakValue = np.zeros(len(transitionPos))
        peakPos = np.zeros(len(transitionPos))
        
        for i, pos in enumerate(transitionPos):
            segment = data[pos-50:pos+51]
            peakValue[i] = np.max(segment)
            peakPos[i] = pos - 50 + np.argmax(segment)
    else:
        return [], []
    
    avg = np.mean(data)
    maxData = np.max(data)
    thresh = avg + 0.8 * (maxData - avg)
    
    mask = peakValue > thresh
    peakValue = peakValue[mask]
    peakPos = peakPos[mask]
    
    return peakValue, peakPos


def h_computeSpeed(time, data, voltageRange, threadmill_diameter=16):
    deltaVoltage = voltageRange[1] - voltageRange[0]
    thresh = 3/5 * deltaVoltage
    
    diffData = np.diff(data, axis=0)
    I = np.abs(diffData) > thresh
    
    data = data.copy()
    for j in range(data.shape[1]):
        if np.any(I[:, j]):
            ind = np.where(I[:, j])[0]
            for i in ind:
                if diffData[i, j] < thresh:
                    data[i+1:, j] = data[i+1:, j] + deltaVoltage
                elif diffData[i, j] > thresh:
                    data[i+1:, j] = data[i+1:, j] - deltaVoltage
    
    dataInDegree = (data / deltaVoltage) * 360
    
    deltaDegree = np.mean(dataInDegree[-11:, :], axis=0) - np.mean(dataInDegree[:11, :], axis=0)
    
    I1 = deltaDegree > 200
    I2 = deltaDegree < -200
    deltaDegree[I1] = deltaDegree[I1] - 360
    deltaDegree[I2] = deltaDegree[I2] + 360
    
    duration = np.mean(time[-11:]) - np.mean(time[:11])
    speed = deltaDegree / duration
    
    diameter = threadmill_diameter
    speed2 = speed / 360 * diameter * np.pi
    
    return speed2


def load_running_data(running_data_path):
    """Load running data from a .ast2 file and return the running data and corresponding time values."""
    header, data = h_AST2_readData(running_data_path)
    activeChIDs = header['activeChIDs']
    if isinstance(activeChIDs, list):
        activeChID = activeChIDs[0]
    elif isinstance(activeChIDs, int):
        activeChID = activeChIDs
    else:
        raise ValueError("Invalid type for activeChIDs in header")
    running_data = h_AST2_raw2Speed(data[activeChID], header, voltageRange=None)
    speed_data = running_data['speed']
    raw_time = running_data['timestamps']
    sample_rate = header['inputRate'] / header['saveEvery']
    return speed_data, raw_time, int(sample_rate)


def align_data(dff, intensity, running_data_path, real_time_xml, episode_file, experiment_xml_file, event_path, timestamp_path):
    """Align running data with imaging data based on trigger signals and return the aligned running data, time, and imaging frame rate."""
    running_data, raw_time, running_sample_rate = load_running_data(running_data_path)
    running_relative_time = (raw_time - raw_time[0])
    
    with h5py.File(episode_file, 'r') as h5f:
        running_sync = np.array(h5f['/AI/Runningdata'])
        frame_out = np.array(h5f['/DI/FrameOut'])
        trigger_signal = np.array(h5f['/DI/Triggersignal'])
    
    tree = ET.parse(real_time_xml)
    root = tree.getroot()
    
    daq = root.find(".//AcquireBoard[@active='1']")
    if daq is None:
        raise RuntimeError("No active DAQ board found in ThorRealTimeDataSettings.xml")  

    sr_node = daq.find(".//SampleRate[@enable='1']")
    if sr_node is None:
        raise RuntimeError("No enabled <SampleRate> found in active DAQ board")

    sample_rate = float(sr_node.get("rate"))

    tree = ET.parse(experiment_xml_file)
    root = tree.getroot()    
    lsm = root.find('.//LSM[@name="ResonanceGalvo"]')
    
    if lsm is not None:
        frame_rate = float(lsm.get('frameRate'))
        average_num = int(lsm.get('averageNum'))
    else:
        print("LSM cannot find!", "WARNING")
        frame_rate = 10.0
        average_num = 1
    
    streaming = root.find('.//Streaming[@enable="1"]')
    if streaming is not None:
        frames = int(streaming.get('frames'))
    else:
        print("Streaming cannot find!", "WARNING")
        frames = 1000
        
    running_onset_idx = np.where(trigger_signal > 0.5)[0]
    imaging_onset_idx = np.where(frame_out > 0.5)[0]
    
    if len(running_onset_idx) > 0 and len(imaging_onset_idx) > 0:
        running_onset = running_onset_idx[0]
        diff = np.diff(imaging_onset_idx)
        breaks = np.where(diff > 100)
        imaging_session = []
        for i in range(len(breaks[0]) + 1):
            if i == 0:
                start = imaging_onset_idx[0]
            else:
                start = imaging_onset_idx[breaks[0][i-1] + 1]
            
            if i == len(breaks[0]) :
                end = imaging_onset_idx[-1]
            else:
                end = imaging_onset_idx[breaks[0][i]]
                
            imaging_session.append((start, end))
            
        for i, (start, end) in enumerate(imaging_session):
            duration = (end - start) / sample_rate
            threshold = 0.005
            imaging_time = frames / (frame_rate / average_num)
            if abs(duration - imaging_time) < threshold*imaging_time:
                imaging_onset = start
                relative_time1 = (running_onset - imaging_onset) / sample_rate
                break
    else:
        relative_time1 = 0.0
        print("Trigger signals not found, using zero offset", "WARNING")
    
    timestamps = pd.read_csv(timestamp_path)
    exp_start = timestamps[(timestamps['Device'] == 'Experiment') &
                        (timestamps['Action'] == 'Start')]['Timestamp'].values
    speed_start = timestamps[(timestamps['Device'] == 'Speed Sensor') &
                        (timestamps['Action'] == 'Start')]['Timestamp'].values
    speed_end = timestamps[(timestamps['Device'] == 'Speed Sensor') &
                        (timestamps['Action'] == 'End')]['Timestamp'].values
    
    event_data = pd.read_csv(event_path)
    start_offset = event_data['start_time'].iloc[0]
    event_data['start_time'] = event_data['start_time'] - start_offset
    event_data['end_time'] = event_data['end_time'] - start_offset
    
    if len(exp_start) > 0 and len(speed_start) > 0:
        relative_time2 = exp_start[0] - speed_start[0]
        duration = speed_end[0] - speed_start[0]
    
    running_time = running_relative_time- relative_time2
    imaging_time = np.arange(frames) / frame_rate * average_num - relative_time1 - relative_time2

    valid_mask_running = (running_time >= 0) & (running_time <= duration)
    aligned_time_running = running_time[valid_mask_running]
    aligned_running_data = running_data[valid_mask_running]
    valid_mask_imaging = (imaging_time >= 0) & (imaging_time <= duration)
    aligned_time_imaging = imaging_time[valid_mask_imaging]
    aligned_intensity = intensity[:, valid_mask_imaging]
    aligned_dff = dff[:, valid_mask_imaging]

    plt.figure()
    plt.plot(frame_out, color="#3CFF00", label='frame out signal')
    plt.plot(trigger_signal, color="#0004FF", label='trigger signal')
    plt.axvline(running_onset, color='#000000', label='running onset')
    plt.axvline(imaging_onset, color="#FF0000", label='imaging onset')
    plt.legend(loc='upper left')
    plt.show()

    return aligned_running_data, aligned_time_running, running_sample_rate, aligned_intensity, aligned_dff, frame_rate / average_num, event_data


def load_and_align_data(intensitys, dffs, groups_file):
    groups = pd.read_csv(groups_file)
    groups_list = groups.values.tolist()
    running_datas = []
    intensitys_aligned = []
    dffs_aligned = []
    imaging_frame_rates = []
    running_sample_rates = []
    events = []
    for i, group in enumerate(groups_list):
        folder_running_datas = []
        folder_intensitys_aligned = []
        folder_dffs_aligned  = []
        folder_imaging_frame_rate = []
        folder_running_sample_rate = []
        folder_events = []
        for j, session in enumerate(group):
            running_data_dir = os.path.dirname(os.path.dirname(session))
            element2 = os.listdir(running_data_dir)
            for root, dirs, files in os.walk(running_data_dir):
                if 'SyncData' in root:
                    sync_dir = root
                    break
                
            if sync_dir:
                real_time_xml = os.path.join(sync_dir, 'ThorRealTimeDataSettings.xml')
                Episodefile = os.path.join(sync_dir, "Episode_0000.h5")
            
            for root, dirs, files in os.walk(running_data_dir):
                if 'Experiment.xml' in files:
                    experiment_xml_file = os.path.join(root, 'Experiment.xml')
                    break
            
            running_data_path = os.path.join(running_data_dir, [f for f in element2 if f.endswith('.ast2')][0])
            event_path = os.path.join(running_data_dir, [f for f in element2 if f.endswith('events.csv')][0])
            timestamp_path = os.path.join(running_data_dir, [f for f in element2 if f.endswith('timestamps.csv') and 'cam' not in f][0])
            
            aligned_running, aligned_time_running, running_sample_rate, aligned_intensity, aligned_dff, imaging_frame_rate, event_data = align_data(
                dff=dffs[i][j],
                intensity=intensitys[i][j],
                running_data_path=running_data_path,
                real_time_xml=real_time_xml,
                episode_file=Episodefile,
                experiment_xml_file=experiment_xml_file,
                event_path=event_path,
                timestamp_path=timestamp_path
            )
            
            folder_running_datas.append((aligned_running, aligned_time_running))
            folder_intensitys_aligned.append(aligned_intensity)
            folder_dffs_aligned.append(aligned_dff)
            folder_imaging_frame_rate.append(imaging_frame_rate)
            folder_running_sample_rate.append(running_sample_rate)
            folder_events.append(event_data)
        running_datas.append(folder_running_datas)
        intensitys_aligned.append(folder_intensitys_aligned)
        dffs_aligned.append(folder_dffs_aligned)
        imaging_frame_rates.append(folder_imaging_frame_rate)
        running_sample_rates.append(folder_running_sample_rate)
        events.append(folder_events)
        
    return intensitys_aligned, dffs_aligned, running_datas, imaging_frame_rates, running_sample_rates, events