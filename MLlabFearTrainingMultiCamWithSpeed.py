# -*- coding: utf-8 -*-
""" 
Created on Tue Jul 8 19:18:40 2025 

Edit on Mon Jul 21 18:46:08 2025 by pulee.

Preview FPS = 5, Multi FLIR Camera, Opencv Cam Time correct.

Edit on Tue Jul 22 14:04:16 2025 by pulee.

Change timestamp from relative to absolute for event log.
Multi Opencv, HIK Camera, HIK Cam Time correct.

Edit on Wed Jul 23 11:16:48 2025 by pulee.
Opencv Cam(IR) Time incorrect, the id is not unique.

Edit on Sat Sep 6 11:37:30 2025 by pulee.
Add the function of optogenetics stimulation.

Edit on Sun Sep 14 20:11:50 2025(Add for the synchronization of imaging) by pulee.

Send the 5V voltage through ao0 to synchronize the signal of running and imaging.

Record the video start time and running start time, synchronize the signal of behaviour and running.

Edit on Mon Nov 28 16:30:21 2025 by pulee.

Add the function to save the timestamp of each frame into csv file for FLIR HIK and Opencv camera.

Edit on Wed Dec 3 20:33:59 2025 by pulee.

Add the function to save the log of optogenetics stimulation time into csv file.

Edit on Wed Dec 24 09:53:51 2025 by pulee.

Fixed the issue of saving files without optical genetic logs

Edit on Wed Jan 7 14:22:33 2026 by pulee.

Repalce the SpeedEncoderUI module to same as h_AnimalStateTracker2_lei.m

Behaviour and video uncoupled

Edit on Tue Jan 13 15:40:12 2026 by pulee.

Repair the issue of multichannel speed encode

@author: Pulee 
""" 

import os
import sys
import cv2
import csv
import time
import queue
import serial
import ctypes
import PySpin
import threading
import matplotlib
import win32com.client
import serial.tools.list_ports
import numpy as np
import tkinter as tk
import sounddevice as sd
import matplotlib.pyplot as plt
from ctypes import *
from PyDAQmx import *
from queue import Queue
from PyDAQmx import Task
from datetime import datetime
from PIL import Image, ImageTk, ImageOps
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PyDAQmx.DAQmxTypes import DAQmxEveryNSamplesEventCallbackPtr

sys.path.append(os.getenv('MVCAM_COMMON_RUNENV') + "/Samples/Python/MvImport")
from CameraParams_header import *
from MvErrorDefine_const import *
from MvCameraControl_class import *

# Set matplotlib backend for Tkinter
matplotlib.use('TkAgg')

# Define pixel type constant
PixelType_Mono8 = 17301505

class FLIRCamera:
    """Class for controlling FLIR cameras using PySpin library"""
    def __init__(self, root, arduino=None, camera_id=0, device_index=0):
        # Initialize camera parameters and settings
        self.root = root
        self.arduino = arduino
        self.camera_id = camera_id
        self.device_index = device_index
        self.display_name = f"Camera {camera_id+1} (FLIR)"

        # Default camera parameters
        self.width = 1280
        self.height = 1024
        self.sample_rate = 120.0
        self.record_rate = 30.0
        self.exposure_time = 1000.0

        # Camera state flags
        self.previewing = False
        self.recording = False
        self.running = False
        self.preview_callback = None

        # Thread synchronization locks
        self.preview_lock = threading.Lock()
        self.record_lock = threading.Lock()
        
        # Thread handlers
        self.acquisition_thread = None
        self.record_thread = None
        
        # FPS tracking
        self.current_fps = 0.0
        self._frame_count = 0
        self._fps_last_time = time.time()

        # Timestamp logging
        self.frame_timestamps = []
        self.timestamp_file = None
        self.timestamp_writer = None

        # Initialize FLIR camera system
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() == 0:
            self.system.ReleaseInstance()
            raise RuntimeError("No FLIR camera detected.")
        
        # Validate device index
        if self.device_index >= self.cam_list.GetSize():
            raise RuntimeError(f"FLIR camera index {self.device_index} not available")
            
        self.cam = self.cam_list[self.device_index]
        if not self.cam.IsInitialized():
            try:
                self.cam.Init()
            except Exception as e:
                print(f"[ERROR] Failed to initialize camera {self.device_index}: {e}")
                raise
        self.nodemap = self.cam.GetNodeMap()

        # Configure camera
        self.open()

        # Frame processing queues
        self.display_queue = queue.Queue(maxsize=1)
        self.record_frame = None
        self.frame_queue = Queue(maxsize=1000)
        # Start display worker thread
        threading.Thread(target=self._display_worker, daemon=True).start()
        
    @staticmethod
    def list_devices():
        """List all available FLIR cameras"""
        try:
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()
            num_cams = cam_list.GetSize()
            devices = []
            for i in range(num_cams):
                cam = cam_list.GetByIndex(i)
                if cam.IsStreaming():
                    cam.EndAcquisition()
                nodemap = cam.GetTLDeviceNodeMap()
                node_device_name = PySpin.CStringPtr(nodemap.GetNode('DeviceModelName'))
                if PySpin.IsAvailable(node_device_name) and PySpin.IsReadable(node_device_name):
                    device_name = node_device_name.ToString()
                else:
                    device_name = f"FLIR Camera {i+1}"
                devices.append(device_name)
                cam.DeInit()
                cam = None
            cam_list.Clear()
            system.ReleaseInstance()
            return devices
        except Exception as e:
            print(f"Error listing FLIR devices: {e}")
            return []

    def open(self):
        """Configure camera settings"""
        def set_node(name, value):
            # Helper function to set float node values
            node = PySpin.CFloatPtr(self.nodemap.GetNode(name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                node.SetValue(min(value, node.GetMax()))

        def set_enum(name, entry_name):
            # Helper function to set enumeration values
            node = PySpin.CEnumerationPtr(self.nodemap.GetNode(name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                entry = node.GetEntryByName(entry_name)
                if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                    node.SetIntValue(entry.GetValue())

        def set_int(name, value):
            # Helper function to set integer values
            node = PySpin.CIntegerPtr(self.nodemap.GetNode(name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                node.SetValue(min(value, node.GetMax()))

        # Apply camera settings
        set_int("OffsetX", 0)
        set_int("OffsetY", 0)
        set_int("Width", self.width)
        set_int("Height", self.height)
        set_enum("ExposureAuto", "Off")
        set_node("ExposureTime", self.exposure_time)
        set_node("AcquisitionFrameRate", self.sample_rate)
        # Enable frame rate control
        fps_enable = PySpin.CBooleanPtr(self.nodemap.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsAvailable(fps_enable) and PySpin.IsWritable(fps_enable):
            fps_enable.SetValue(True)

    def set_parameters(self, width, height, exposure_time, sample_rate, record_rate):
        """Update camera parameters"""
        self.width = width
        self.height = height
        self.exposure_time = exposure_time
        self.sample_rate = sample_rate
        self.record_rate = record_rate

    def start_preview(self, callback):
        """Start camera preview"""
        with self.preview_lock:
            # Ensure camera is initialized
            if not self.cam.IsInitialized():
                print("[ERROR] Camera is not initialized. Trying to initialize...")
                try:
                    self.cam.Init()
                except Exception as e:
                    print(f"[ERROR] Failed to initialize camera: {e}")
                    return
                if not self.cam.IsInitialized():
                    print("[ERROR] Camera initialization failed. Cannot start preview.")
                    return
                    
            # Start acquisition if not already streaming
            if not self.cam.IsStreaming():
                try:
                    self.cam.BeginAcquisition()
                except Exception as e:
                    print(f"[ERROR] Failed to start acquisition: {e}")
                    return

            # Set preview state and callback
            self.previewing = True
            self.preview_callback = callback
            # Start acquisition thread if not running
            if not self.running:
                self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
                self.acquisition_thread.start()

    def stop_preview(self):
        """Stop camera preview"""
        with self.preview_lock:
            self.previewing = False

    def start_record(self, save_path):
        """Start video recording"""
        with self.record_lock:
            # Ensure camera is initialized
            if not self.cam.IsInitialized():
                print("[ERROR] Camera is not initialized. Trying to initialize...")
                try:
                    self.cam.Init()
                except Exception as e:
                    print(f"[ERROR] Failed to initialize camera: {e}")
                    return
                if not self.cam.IsInitialized():
                    print("[ERROR] Camera initialization failed. Cannot start recording.")
                    return
                    
            # Start acquisition if not already streaming
            if not self.cam.IsStreaming():
                try:
                    self.cam.BeginAcquisition()
                except Exception as e:
                    print(f"[ERROR] Failed to start acquisition: {e}")
                    return

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(save_path, fourcc, self.record_rate, (self.width, self.height))
            if not self.out.isOpened():
                print("[ERROR] Cannot open video writer")
                return
            
            # Initialize timestamp logging
            timestamp_path = save_path.replace('.avi', '_timestamps.csv')
            try:
                self.timestamp_file = open(timestamp_path, 'w', newline='')
                self.timestamp_writer = csv.writer(self.timestamp_file)
                self.timestamp_writer.writerow(['Frame_Index', 'Timestamp', 'Relative_Time'])
                self.frame_timestamps = []
                print(f"Timestamp file created: {timestamp_path}")
            except Exception as e:
                print(f"[ERROR] Failed to create timestamp file: {e}")

            # Set recording state and mark start
            self.recording = True
            self.running = True
            print(f"Recording start for camera {self.camera_id}")
            self.start_time = time.time()
            self.mark_ttl_on()  # Send TTL signal

            # Start acquisition thread if not running
            if not self.acquisition_thread or not self.acquisition_thread.is_alive():
                self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
                self.acquisition_thread.start()
                
            # Start recording thread
            self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
            self.record_thread.start()

        return self.start_time

    def _acquisition_loop(self):
        """Main acquisition loop for capturing frames"""
        try:
            with self.preview_lock:
                # Reinitialize camera if needed
                if not self.cam.IsInitialized():
                    print("[ERROR] Camera is not initialized. Trying to re-initialize...")
                    try:
                        self.cam.Init()
                    except Exception as e:
                        print(f"[ERROR] Failed to re-initialize camera: {e}")
                        return
                # Start acquisition if not streaming
                if not self.cam.IsStreaming():
                    self.cam.BeginAcquisition()
            self.running = True
            
            # Timing parameters
            preview_interval = 1.0 / 5.0  # Preview at 5 FPS
            last_preview_time = time.time()
            record_interval = 1.0 / self.record_rate

            next_record_time = time.time()
            
            frame_index = 0

            # Main acquisition loop
            while self.previewing or self.recording:
                # Get next image
                image_result = self.cam.GetNextImage(1000)
                if image_result.IsIncomplete():
                    continue

                # Process image
                img = image_result.GetNDArray()
                image_result.Release()
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

                # Calculate FPS
                self._frame_count += 1
                if time.time() - self._fps_last_time >= 1.0:
                    self.current_fps = self._frame_count / (time.time() - self._fps_last_time)
                    self._frame_count = 0
                    self._fps_last_time = time.time()

                now = time.time()

                # Handle preview frames
                with self.preview_lock:
                    if self.previewing and now - last_preview_time >= preview_interval:
                        last_preview_time = now
                        if not self.display_queue.full():
                            self.display_queue.put(frame)

                # Handle recording frames
                with self.record_lock:
                    if self.recording:
                        if now >= next_record_time:
                            next_record_time += record_interval
                            if not self.frame_queue.full():
                                self.frame_queue.put((time.time(), frame.copy()))

                                # Log timestamp
                                current_timestamp = time.time()
                                relative_time = current_timestamp - self.start_time
                                self.frame_timestamps.append((frame_index, current_timestamp, relative_time))
                                
                                if self.timestamp_writer:
                                    self.timestamp_writer.writerow([frame_index, f"{current_timestamp:.6f}", f"{relative_time:.6f}"])
                                
                                frame_index += 1

        except Exception as e:
            print("[ERROR] Acquisition failed:", e)
        finally:
            # Cleanup
            self.running = False
            try:
                with self.preview_lock:
                    if self.cam.IsStreaming():
                        self.cam.EndAcquisition()
            except:
                pass

    def record_loop(self):
        """Process frames for recording"""
        while self.recording:
            try:
                # Get frame from queue and write to video
                timestamp, frame = self.frame_queue.get(timeout=0.5)
                self.out.write(frame)
            except queue.Empty:
                time.sleep(0.01)

    def _display_worker(self):
        """Worker thread for displaying preview frames"""
        while True:
            try:
                # Get frame from display queue and convert to PIL format
                frame = self.display_queue.get()
                if self.preview_callback:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # Schedule callback in main thread
                    self.root.after(0, lambda img=pil_img: self.preview_callback(img, self.camera_id))
            except Exception as e:
                print("[ERROR] Display thread:", e)

    def stop_record(self):
        """Stop video recording"""
        print(f"Recording stopped for camera {self.camera_id}")
        end_time = time.time
        if self.recording:
            self.mark_ttl_off()  # Send TTL off signal

        with self.record_lock:
            # Update state flags
            self.recording = False
            self.running = False
            
            # Stop recording thread
            if self.record_thread and self.record_thread.is_alive():
                self.record_thread.join(timeout=1.0)
                
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break

        # Release video writer
        if hasattr(self, "out") and self.out:
            self.out.release()

        # Close timestamp file
        if self.timestamp_file:
            try:
                self.timestamp_file.close()
                print(f"Timestamp file saved for camera {self.camera_id}")
            except Exception as e:
                print(f"[ERROR] Failed to close timestamp file: {e}")
            finally:
                self.timestamp_file = None
                self.timestamp_writer = None

        return end_time

    def close(self):
        """Cleanup and release camera resources"""
        self.stop_preview()
        self.stop_record()
        
        # Stop acquisition thread
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=1.0)

        # Release camera resources
        if hasattr(self, 'cam'):
            try:
                if self.cam.IsStreaming():
                    self.cam.EndAcquisition()
            except:
                pass
            self.cam.DeInit()
            del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()
        print(f"[INFO] FLIR Camera {self.camera_id} successfully closed.")

    def mark_ttl_on(self):
        """Send TTL ON signal to Arduino"""
        self._send_to_arduino('T')
    
    def mark_ttl_off(self):
        """Send TTL OFF signal to Arduino"""
        self._send_to_arduino('t')
    
    def _send_to_arduino(self, command_char):
        """Send command to Arduino via serial"""
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command_char.encode())
                print(f"Sent to Arduino: {command_char}")
        except Exception as e:
            print(f"Serial communication error: {e}")

class HKCamera:
    def __init__(self, root, arduino=None, camera_id=0, device_index=0):
        self.root = root
        self.arduino = arduino
        self.camera_id = camera_id
        self.device_index = device_index
        self.display_name = f"Camera {camera_id+1} (HIK)"

        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.tlayer_type = MV_USB_DEVICE

        self.recording = False
        self.previewing = False
        self.running = False
        self.preview_callback = None
        self.save_path = None

        self.pixel_type = PixelType_Mono8
        self.last_frame_data = None
        self.frame_counter = 0
        self.acq_queue = queue.Queue(maxsize=200)
        self.acqDuration = []
        self.recDuration = []

        self.width = 1280
        self.height = 1024
        self.record_rate = 30.0
        self.sample_rate = 90.0
        self.exposure_time = 1000.0
        
        self.start_time = None
        self.end_time = None
        self.current_fps = 0.0
        self.fps_counter = 0
        self.last_fps_time = time.time()

        self.frame_timestamps = []
        self.timestamp_file = None
        self.timestamp_writer = None

    @staticmethod
    def list_devices():
        """List all available HIK cameras"""
        try:
            device_list = MV_CC_DEVICE_INFO_LIST()
            tlayer_type = MV_USB_DEVICE
            ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
            if ret != MV_OK:
                return []
            
            devices = []
            for i in range(device_list.nDeviceNum):
                dev_info = device_list.pDeviceInfo[i].contents
                if dev_info.nTLayerType == MV_USB_DEVICE:
                    # model_name = ""
                    serial_number = ""
                    # if dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    #     model_name = bytes(dev_info.SpecialInfo.stUsb3VInfo.chModelName).decode('utf-8').rstrip('\x00')
                    # else:
                    #     model_name = f"HIK Camera {i+1}"
                    if dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                        serial_number = bytes(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber).decode('utf-8').rstrip('\x00')
                    else:
                        serial_number = f"SN_{i+1}"

                    devices.append(serial_number)
            return devices
        except Exception as e:
            print(f"Error listing HIK devices: {e}")
            return []

    def open(self):
        ret = MvCamera.MV_CC_EnumDevices(self.tlayer_type, self.device_list)
        if ret != MV_OK or self.device_list.nDeviceNum == 0:
            raise RuntimeError("No camera found.")
    
        if self.device_index >= self.device_list.nDeviceNum:
            raise RuntimeError(f"Camera index {self.device_index} not available")
            
        dev_info = self.device_list.pDeviceInfo[self.device_index].contents
        self.cam.MV_CC_CreateHandle(dev_info)
        self.cam.MV_CC_OpenDevice()
    
        self.cam.MV_CC_SetEnumValue("PixelFormat", self.pixel_type)

        self.cam.MV_CC_SetIntValue("Width", self.width)
        self.cam.MV_CC_SetIntValue("Height", self.height)

        self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.sample_rate)
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetGrabStrategy(MV_GrabStrategy_LatestImages)
        self.cam.MV_CC_StartGrabbing()
        print(f"[INFO] Opened camera {self.camera_id} with resolution {self.width}x{self.height} @ {self.record_rate}fps")

    def close(self):
        try:
            if hasattr(self, 'cam'):
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
                print(f"[INFO] HIK Camera {self.camera_id} successfully closed.")
        except Exception as e:
            print(f"[WARN] Error closing camera {self.camera_id}: {e}")
    
    def set_parameters(self, width, height, exposure_time, sample_rate, record_rate):
        self.width = width
        self.height = height
        self.exposure_time = exposure_time
        self.sample_rate = sample_rate
        self.record_rate = record_rate

    def start_preview(self, callback):
        if self.previewing:
            print(f"[WARN] Camera {self.camera_id} preview already running.")
            return
        try:
            if not hasattr(self, 'cam') or not self.cam:
                self.open()
                
            self.previewing = True
            self.preview_callback = callback
            threading.Thread(target=self._preview_loop, daemon=True).start()
        except Exception as e:
            print(f"[ERROR] Camera {self.camera_id} preview failed: {e}")

    def stop_preview(self):
        self.previewing = False

    def _preview_loop(self):
        stFrame = MV_FRAME_OUT()
        preview_interval = 1.0 / 5.0
        last_preview_time = time.time()
        
        while self.previewing:
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == MV_OK:
                buf = string_at(stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen)
                img_np = np.frombuffer(buf, dtype=np.uint8).reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))
                pil_img = Image.fromarray(img_np)

                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = time.time()
                
                self.cam.MV_CC_FreeImageBuffer(stFrame)
                
                now = time.time()
                if now - last_preview_time >= preview_interval:
                    last_preview_time = now
                    if self.preview_callback:
                        self.root.after(0, lambda img=pil_img: self.preview_callback(img, self.camera_id))
            else:
                time.sleep(0.01)

    def start_record(self, save_path):
        self.running = True
        self.recording = True
        self.save_path = save_path
        self.frame_counter = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()

        param = MV_CC_RECORD_PARAM()
        param.enPixelType = self.pixel_type
        param.nWidth = self.width
        param.nHeight = self.height
        param.fFrameRate = self.record_rate
        param.nBitRate = 8192
        param.enRecordFmtType = 1
        param.strFilePath = c_char_p(save_path.encode('utf-8'))
        ret = self.cam.MV_CC_StartRecord(param)
        if ret != MV_OK:
            print(f"[ERROR] Failed to start recording for path {save_path}: {ret}")

        timestamp_path = save_path.replace('.avi', '_timestamps.csv')
        try:
            self.timestamp_file = open(timestamp_path, 'w', newline='')
            self.timestamp_writer = csv.writer(self.timestamp_file)
            self.timestamp_writer.writerow(['Frame_Index', 'Timestamp', 'Relative_Time'])
            self.frame_timestamps = []
            print(f"Timestamp file created: {timestamp_path}")
        except Exception as e:
            print(f"[ERROR] Failed to create timestamp file: {e}")

        print(f"Recording start for camera {self.camera_id}")
        self.start_time = time.time()
        self.mark_ttl_on()
        
        threading.Thread(target=self._acquisition_loop, daemon=True).start()
        threading.Thread(target=self._record_loop, daemon=True).start()
        return self.start_time

    def _acquisition_loop(self):
        stFrame = MV_FRAME_OUT()
        while self.running:
            t0 = time.time()
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == MV_OK:
                buf = string_at(stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen)
                self.last_frame_data = buf
                h = stFrame.stFrameInfo.nHeight
                w = stFrame.stFrameInfo.nWidth
                expected_size = h * w

                img_np = np.frombuffer(buf, dtype=np.uint8)

                if img_np.size >= expected_size:
                    img_np = img_np[:expected_size].reshape((h, w))
                else:
                    print(f"[ERROR] Received frame size {img_np.size}, expected {expected_size}. Skipping frame.")
                    continue

                self.cam.MV_CC_FreeImageBuffer(stFrame)

                while not self.acq_queue.empty():
                    try:
                        self.acq_queue.get_nowait()
                    except queue.Empty:
                        break

                self.acq_queue.put((img_np.copy(), time.time()))

                self.acqDuration.append(time.time() - t0)
            else:
                time.sleep(0.005)
            
    def _record_loop(self):
        stInput = MV_CC_INPUT_FRAME_INFO()
        self.frame_counter = 0
        start_time = time.time()
        target_interval_t = 1.0 / self.record_rate  # Target interval for recording frame rate

        while self.recording:
            record_loop_start = time.time()
    
            if self.last_frame_data is None:
                time.sleep(0.01)
                continue

            if not self.acq_queue.empty():
                img_np, _ = self.acq_queue.get()
                # Convert to bytes for SDK input
                self.last_frame_data = img_np.tobytes()
    
            stInput.pData = cast(c_char_p(self.last_frame_data), POINTER(c_ubyte))
            stInput.nDataLen = len(self.last_frame_data)
            ret = self.cam.MV_CC_InputOneFrame(stInput)
            if ret != MV_OK:
                print(f"[ERROR] Failed to input frame: {ret}")

            current_timestamp = time.time()
            relative_time = current_timestamp - start_time
            self.frame_timestamps.append((self.frame_counter, current_timestamp, relative_time))
            
            if self.timestamp_writer:
                self.timestamp_writer.writerow([self.frame_counter, f"{current_timestamp:.6f}", f"{relative_time:.6f}"])

            self.frame_counter += 1

            self.fps_counter += 1
            if time.time() - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = time.time()
    
            target_interval = self.frame_counter * target_interval_t - (time.time() - start_time)
            if target_interval > 0:
                time.sleep(target_interval)
            record_loop_end = time.time()
            self.recDuration.append(record_loop_end - record_loop_start)

    def stop_record(self):
        self.end_time = time.time()
        if self.recording is True:
            self.mark_ttl_off()

        self.running = False
        self.recording = False
        self.cam.MV_CC_StopRecord()

        if self.timestamp_file:
            try:
                self.timestamp_file.close()
                print(f"Timestamp file saved for camera {self.camera_id}")
            except Exception as e:
                print(f"[ERROR] Failed to close timestamp file: {e}")
            finally:
                self.timestamp_file = None
                self.timestamp_writer = None

        print(f"Recording stopped for camera {self.camera_id}")
        return self.end_time
    
    def mark_ttl_on(self):
        self._send_to_arduino('T')
    
    def mark_ttl_off(self):
        self._send_to_arduino('t')
    
    def _send_to_arduino(self, command_char):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command_char.encode())
                print(f"Sent to Arduino: {command_char}")
        except Exception as e:
            print(f"Serial communication error: {e}")

class OpenCVCamera:
    def __init__(self, root, arduino=None, camera_id=0, device_index=0, num_channels=1):
        self.root = root
        self.arduino = arduino
        self.camera_id = camera_id
        self.display_name = f"Camera {camera_id+1} (OpenCV)"
        self.num_channels = num_channels
        self.device_index = device_index
        
        self.capture = None
        self.video_writers = [None] * num_channels
        self.previewing = False
        self.recording = False
        self.frame_counter = 0
        self.start_time = None
        self.end_time = None
        self.current_fps = 0.0
        self.fps_counter = 0
        self.last_fps_time = time.time()  # ???
        
        self.width = 640
        self.height = 480
        self.fps = 30.0

        self.frame_timestamps = []
        self.timestamp_files = [None] * num_channels
        self.timestamp_writers = [None] * num_channels
        self.frame_indices = [0] * num_channels

        self.recording_threads = [None] * num_channels
        self.frame_queues = [queue.Queue(maxsize=1000) for _ in range(num_channels)]
        
    def open(self):
        self.capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.device_index}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        print(f"[INFO] Opened OpenCV camera {self.camera_id} with resolution {self.width}x{self.height} @ {self.fps:.2f}fps")

    def close(self):
        if self.capture:
            self.capture.release()
        for video_writer in self.video_writers:
            if video_writer:
                video_writer.release()
        self.capture = None
        self.video_writers = [None] * self.num_channels
        print(f"[INFO] OpenCV Camera {self.camera_id} successfully closed.")

    def set_parameters(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps

    def start_preview(self, callback):
        if self.previewing:
            print(f"[WARN] Camera {self.camera_id} preview already running.")
            return
        self.previewing = True
        self.preview_callback = callback
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def stop_preview(self):
        self.previewing = False

    def _preview_loop(self):
        preview_interval = 1.0 / 5.0
        last_preview_time = time.time()
        
        while self.previewing:
            if self.capture is None or not self.capture.isOpened():
                try:
                    self.capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                    if not self.capture.isOpened():
                        print(f"[ERROR] Camera {self.camera_id} reopen failed.")
                        time.sleep(1)
                        continue
                    
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.capture.set(cv2.CAP_PROP_FPS, self.fps)
                except Exception as e:
                    print(f"[ERROR] Camera {self.camera_id} reopen failed: {e}")
                    time.sleep(1)
                    continue

            ret, frame = self.capture.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = time.time()
                
                now = time.time()
                if now - last_preview_time >= preview_interval:
                    last_preview_time = now
                    if self.preview_callback:
                        self.root.after(0, lambda img=pil_img: self.preview_callback(img, self.camera_id))
                
                if self.recording:
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    for i in range(self.num_channels):
                        if not self.frame_queues[i].full():
                            self.frame_queues[i].put(frame.copy())
            else:
                print(f"[WARN] Camera {self.camera_id} read frame failed. Reopening...")
                self.capture.release()
                self.capture = None
                time.sleep(0.01)

    def start_record(self, save_paths):
        if not self.capture or not self.capture.isOpened():
            raise RuntimeError("Camera not opened")
            
        if not isinstance(save_paths, list):
            save_paths = [save_paths]

        self.recording = True
        self.save_paths = save_paths
        self.frame_counter = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for i, save_path in enumerate(save_paths):
            self.video_writers[i] = cv2.VideoWriter(save_path, fourcc, self.fps, (self.width, self.height))
            if self.video_writers[i] is None or not self.video_writers[i].isOpened():
                raise RuntimeError(
                    f"VideoWriter failed. Path: {save_path}, "
                    f"FPS: {self.fps}, Size: ({self.width}, {self.height})"
                )
            
        for i, save_path in enumerate(save_paths):
            timestamp_path = save_path.replace('.mp4', '_timestamps.csv').replace('.avi', '_timestamps.csv')
            try:
                self.timestamp_files[i] = open(timestamp_path, 'w', newline='')
                self.timestamp_writers[i] = csv.writer(self.timestamp_files[i])
                self.timestamp_writers[i].writerow(['Frame_Index', 'Timestamp', 'Relative_Time'])
                self.frame_indices[i] = 0
                print(f"Timestamp file created: {timestamp_path}")
            except Exception as e:
                print(f"[ERROR] Failed to create timestamp file for channel {i}: {e}")

        self.frame_timestamps = []

        print(f"Recording start for camera {self.camera_id}")
        self.start_time = time.time()
        self.mark_ttl_on()
        for i in range(self.num_channels):
            self.recording_threads[i] = threading.Thread(target=self._record_loop, args=(i,), daemon=True)
            self.recording_threads[i].start()
        return self.start_time

    def _record_loop(self, channel):
        while self.recording:
            try:
                frame = self.frame_queues[channel].get(timeout=0.1)
                
                if self.video_writers[channel] is None or not self.video_writers[channel].isOpened():
                    print("[WARN] Video writer not available, skipping frame")
                    break
                    
                try:
                    self.video_writers[channel].write(frame)

                    current_timestamp = time.time()
                    relative_time = current_timestamp - self.start_time
                    
                    if self.timestamp_writers[channel]:
                        self.timestamp_writers[channel].writerow([
                            self.frame_indices[channel], 
                            f"{current_timestamp:.6f}", 
                            f"{relative_time:.6f}"
                        ])
                    
                    self.frame_indices[channel] += 1

                except Exception as e:
                    print(f"[ERROR] Failed to write frame: {e}")
                    break
                    
                self.fps_counter += 1
            except queue.Empty:
                time.sleep(0.01)

    def stop_record(self):
        end_time = time.time()
        if self.recording is True:
            self.mark_ttl_off()

        self.recording = False
        
        for i in range(self.num_channels):
            if self.recording_threads[i] and self.recording_threads[i].is_alive():
                self.recording_threads[i].join(timeout=1.0)
            if self.video_writers[i]:
                try:
                    self.video_writers[i].release()
                except Exception as e:
                    print(f"[WARN] Error releasing video writer: {e}")
                finally:
                    self.video_writers[i] = None
            while not self.frame_queues[i].empty():
                try:
                    self.frame_queues[i].get_nowait()
                except:
                    break
        
        for i in range(self.num_channels):
            if self.timestamp_files[i]:
                try:
                    self.timestamp_files[i].close()
                    print(f"Timestamp file saved for camera {self.camera_id}, channel {i}")
                except Exception as e:
                    print(f"[WARN] Error closing timestamp file: {e}")
                finally:
                    self.timestamp_files[i] = None
                    self.timestamp_writers[i] = None

        print(f"Recording stopped for camera {self.camera_id}")
        return end_time
    
    def mark_ttl_on(self):
        self._send_to_arduino('T')
    
    def mark_ttl_off(self):
        self._send_to_arduino('t')
    
    def _send_to_arduino(self, command_char):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command_char.encode())
                print(f"Sent to Arduino: {command_char}")
        except Exception as e:
            print(f"Serial communication error: {e}")

class SpeedEncoderGUI:
    def __init__(self, root, parent, daq_device="Dev2"):
        self.root = root
        self.parent = parent
        self.running = False
        self.task = None
        self.file = None
        self.csv_file = None
        self.csv_writer = None

        self.input_rate = 1000
        self.down_sample = 50
        self.speed_down_sample_factor = 50
        self.device = daq_device
        self.channels = [f'{self.device}/ai0', f'{self.device}/ai1', f'{self.device}/ai2', f'{self.device}/ai3']
        self.num_channels = len(self.channels)
        self.scale = 2 ** 15 / 10  # Scale factor for int16 conversion

        self.buffer_size = 50000
        self.data = np.zeros((self.num_channels, self.buffer_size))
        self.time = np.zeros(self.buffer_size)
        self.current_idx = 0
        self.voltage_range = np.array([[0, 5] for _ in range(self.num_channels)])
        self.speed_data = [[] for _ in range(self.num_channels)]
        self.transition_counts = np.zeros(self.num_channels)
        self._speed_accumulator = [[] for _ in range(self.num_channels)]
        self.first_sample_corrected = [False] * self.num_channels

        # Variables for AST2 file saving
        self.listener_count = 0
        self.start_time = None
        self.csv_frame_index = 0

        self.data_updated = False
        self._update_job = None

        self.enabled_channels = [tk.BooleanVar(value=True) for _ in range(self.num_channels)]
        self.view_mode = [tk.BooleanVar(value=False) for _ in range(self.num_channels)]
        self.auto_scale = [tk.BooleanVar(value=True) for _ in range(self.num_channels)]
        self.x_range = [tk.StringVar(value='') for _ in range(self.num_channels)]
        self.y_range = [tk.StringVar(value='') for _ in range(self.num_channels)]

        self.init_ui()
        self.update_display_loop()
        
    def init_ui(self):
        """Initialize speed sensor UI"""
        self.ch_frame = ttk.LabelFrame(self.parent, text="Channel Controls")
        self.ch_frame.pack(fill=tk.X, padx=5, pady=5)

        for i in range(self.num_channels):
            row_frame = ttk.Frame(self.ch_frame)
            row_frame.pack(fill=tk.X, padx=5, pady=2)

            chk = ttk.Checkbutton(row_frame, text=f"Ch{i}", variable=self.enabled_channels[i])
            chk.pack(side=tk.LEFT, padx=5)

            speed_chk = ttk.Checkbutton(row_frame, text="Show Speed", variable=self.view_mode[i])
            speed_chk.pack(side=tk.LEFT, padx=5)

            auto_chk = ttk.Checkbutton(row_frame, text="Auto Y", variable=self.auto_scale[i])
            auto_chk.pack(side=tk.LEFT, padx=5)

            ttk.Label(row_frame, text="X Lim:").pack(side=tk.LEFT, padx=(10,0))
            x_entry = ttk.Entry(row_frame, textvariable=self.x_range[i], width=10)
            x_entry.pack(side=tk.LEFT)

            ttk.Label(row_frame, text="Y Lim:").pack(side=tk.LEFT, padx=(10,0))
            y_entry = ttk.Entry(row_frame, textvariable=self.y_range[i], width=10)
            y_entry.pack(side=tk.LEFT)

        self.btn_frame = ttk.Frame(self.parent)
        self.btn_frame.pack(fill=tk.X, padx=5, pady=5)

        apply_btn = ttk.Button(self.btn_frame, text="Apply Limits", command=self.apply_axis_limits)
        apply_btn.pack(side=tk.LEFT, padx=5)

        calibrate_btn = ttk.Button(self.btn_frame, text="Auto Calibrate", command=self.auto_calibrate_voltage_range)
        calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="Speed sensor ready")
        self.status_label = ttk.Label(self.parent, textvariable=self.status_var)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        self.ch_frame.pack_forget()
        self.btn_frame.pack_forget()
        self.status_label.pack_forget()
        self.create_speed_plot()
        
    def create_speed_plot(self):
        """Create speed figure"""
        self.plot_frame = ttk.LabelFrame(self.parent, text="Speed Plots")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(self.num_channels, 1, figsize=(8, 6), sharex=True)
        if self.num_channels == 1:
            self.ax = [self.ax]
        
        self.lines = []
        for i, ax in enumerate(self.ax):
            line, = ax.plot([], [], label=f"Channel {i}")
            self.lines.append(line)
            ax.set_xlim(0, 10)
            ax.set_ylim(-10, 10)
            ax.grid(False)
            ax.legend(loc='upper right')
            ax.set_ylabel("Voltage (V)" if not self.view_mode[i].get() else "Speed (°/s)")
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.plot_frame.pack_forget()

    def get_unique_filename(self, path, base):
        """Generate a unique filename for the AST2 output file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(base)
        ext = '.ast2'
        
        filename = f"{name}_AST2_{timestamp}{ext}"
        full_path = os.path.join(path, filename)
        
        counter = 1
        while os.path.exists(full_path):
            filename = f"{name}_AST2_{timestamp}_{counter}{ext}"
            full_path = os.path.join(path, filename)
            counter += 1
            
        return full_path

    def _write_ast2_header(self, file):
        """Write AST2 file header matching MATLAB format"""
        # Get active channel IDs
        active_ch_ids = [i for i, enabled in enumerate(self.enabled_channels) if enabled.get()]
        
        # Create header dictionary matching MATLAB structure
        header_dict = {
            'inputRate': self.input_rate,
            'recordRate': self.input_rate / self.speed_down_sample_factor,
            'device': self.device,
            'acqChIDs': [i for i in range(self.num_channels)],
            'acqInputRange': [-10, 10],
            'saveEvery': self.down_sample,
            'scale': self.scale,
            'voltageRange': self.voltage_range[active_ch_ids].tolist(),
            'activeChIDs': active_ch_ids,
            'startTime': datetime.now().strftime("%d-%b-%Y %H:%M:%S.%f")[:-3]
        }
        
        # Write header in MATLAB-compatible format
        header_str = self._make_header_str(header_dict, 'header')
        header_str += 'header_end\n'
        
        file.write(header_str.encode('utf-8'))

    def _make_header_str(self, data_dict, var_name, indent=0):
        """Convert dictionary to MATLAB-style header string"""
        header_lines = []
        indent_str = ' ' * indent
        
        for key, value in data_dict.items():
            if isinstance(value, dict):
                header_lines.append(f"{indent_str}{var_name}.{key} = struct();")
                header_lines.append(self._make_header_str(value, f"{var_name}.{key}", indent))
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], list):
                    # 2D array
                    array_str = '; '.join([' '.join(map(str, row)) for row in value])
                    header_lines.append(f"{indent_str}{var_name}.{key} = [{array_str}];")
                else:
                    # 1D array
                    array_str = ' '.join(map(str, value))
                    header_lines.append(f"{indent_str}{var_name}.{key} = [{array_str}];")
            elif isinstance(value, str):
                header_lines.append(f"{indent_str}{var_name}.{key} = '{value}';")
            else:
                header_lines.append(f"{indent_str}{var_name}.{key} = {value};")
        
        return '\n'.join(header_lines) + '\n'

    def start_tracking(self, file_path):
        """Start tracking and save AST2 file + CSV file"""
        if self.running:
            return False
            
        self.send_5v_signal()

        # Generate unique AST2 filename
        base_name = os.path.basename(file_path)
        save_dir = os.path.dirname(file_path)
        self.ast2_file_path = self.get_unique_filename(save_dir, base_name)
        
        # Generate CSV filename
        self.csv_file_path = self.ast2_file_path.replace('.ast2', '_speed.csv')
        
        try:
            # Open AST2 file (binary write mode) - for raw voltage data
            self.file = open(self.ast2_file_path, 'wb')
            # Write file header
            self._write_ast2_header(self.file)
            print(f"AST2 file created: {self.ast2_file_path}")
            
            # Open CSV file - for downsampled speed data
            self.csv_file = open(self.csv_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write CSV header
            active_ch_names = [f"Ch{i}_Speed(deg/s)" for i, enabled in enumerate(self.enabled_channels) if enabled.get()]
            self.csv_writer.writerow(['Time(s)'] + active_ch_names)
            print(f"CSV file created: {self.csv_file_path}")
            
        except IOError as e:
            messagebox.showerror("Error", f"Failed to create files: {e}")
            return False
            
        # Initialize DAQ task
        try:
            self.task = Task()
            enabled_count = 0
            for i, ch in enumerate(self.channels):
                if self.enabled_channels[i].get():
                    self.task.CreateAIVoltageChan(
                        ch, "", DAQmx_Val_Cfg_Default, 
                        -10.0, 10.0, DAQmx_Val_Volts, None
                    )
                    enabled_count += 1
                    
            if enabled_count == 0:
                raise ValueError("No channels enabled")
            
            # Configure timing - use input_rate for raw data acquisition
            samples_per_callback = 250  # Fixed callback interval
            self.task.CfgSampClkTiming(
                "", self.input_rate, DAQmx_Val_Rising, 
                DAQmx_Val_ContSamps, 100000
            )
            self.callback = DAQmxEveryNSamplesEventCallbackPtr(self.listener_callback)
            self.task.RegisterEveryNSamplesEvent(
                DAQmx_Val_Acquired_Into_Buffer, 
                samples_per_callback, 
                0, self.callback, None
            )
            self.task.StartTask()
            self.running = True
            self.start_time = time.time()
            self.listener_count = 0
            self.csv_frame_index = 0
            self.update_display_loop()
            self._reset_data()
            
            return True
        except Exception as e:
            messagebox.showerror("DAQ Error", f"Failed to start acquisition: {e}")
            if self.file:
                self.file.close()
                self.file = None
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            return False

    def send_5v_signal(self, duration=0.1):
        """Send 5V synchronization signal"""
        try:
            if not PYDAQMX_AVAILABLE:
                print("PyDAQmx not available, cannot send 5V signal")
                return
                
            task = Task()
            task.CreateAOVoltageChan(f"{self.device}/ao0", "", -10.0, 10.0, DAQmx_Val_Volts, None)
            
            sampling_rate = 1000
            num_samples = int(sampling_rate * duration)
            
            output_data = np.full(num_samples, 5.0, dtype=np.float64)
            
            task.CfgSampClkTiming("", sampling_rate, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, num_samples)
            
            written = ctypes.c_int32()
            task.WriteAnalogF64(num_samples, False, 10.0, DAQmx_Val_GroupByChannel, 
                            output_data, ctypes.byref(written), None)
            
            task.StartTask()
            task.WaitUntilTaskDone(duration + 1.0)
            
            task.StopTask()
            task.ClearTask()

            reset_task = Task()
            reset_task.CreateAOVoltageChan(f"{self.device}/ao0", "", -10.0, 10.0, 10348, None)
            reset_task.StartTask()
            reset_task.WriteAnalogScalarF64(True, 0.001, 0.0, None)
            reset_task.StopTask()
            reset_task.ClearTask()
            
            print(f"Sent 5V signal for {duration} seconds on {self.device}/ao0")
            
        except Exception as e:
            print(f"Error sending 5V signal: {e}")
            try:
                task.StopTask()
                task.ClearTask()
            except:
                pass

    def _reset_data(self):
        """Reset all data structures"""
        self.current_idx = 0
        self.data = np.zeros((self.num_channels, self.buffer_size))
        self.time = np.zeros(self.buffer_size)
        self.speed_data = [[] for _ in range(self.num_channels)]
        self.transition_counts = np.zeros(self.num_channels)
        self._speed_accumulator = [[] for _ in range(self.num_channels)]
        
    def listener_callback(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        """Callback for DAQ data acquisition"""
        try:
            read = ctypes.c_int32()
            active_channels = sum(1 for ch in self.enabled_channels if ch.get())
            buffer = np.zeros((active_channels, number_of_samples))

            self.task.ReadAnalogF64(
                number_of_samples, 10.0, DAQmx_Val_GroupByChannel,
                buffer, buffer.size, ctypes.byref(read), None
            )
            samples = read.value

            self.parent.after(0, lambda: self.handle_data(buffer.copy(), samples))
            return 0
        except Exception as e:
            print(f"Error in callback: {e}")
            return -1

    def handle_data(self, buffer, samples):
        """Process data and save to AST2 file and CSV file"""
        if self.current_idx + samples > self.data.shape[1]:
            new_size = max(self.data.shape[1] * 2, self.current_idx + samples)
            self.data = np.pad(self.data, ((0, 0), (0, new_size - self.data.shape[1])), mode='constant')
            self.time = np.pad(self.time, (0, new_size - len(self.time)), mode='constant')

        start_time = self.time[self.current_idx - 1] + 1 / self.input_rate if self.current_idx else 0
        self.time[self.current_idx:self.current_idx + samples] = np.linspace(
            start_time, start_time + samples / self.input_rate, samples)

        buffer_idx = 0
        for i in range(self.num_channels):
            if self.enabled_channels[i].get():
                self.data[i, self.current_idx:self.current_idx + samples] = buffer[buffer_idx, :samples]
                buffer_idx += 1

        # Save raw voltage data to AST2 file immediately (no downsampling)
        self.save_raw_data_to_ast2(samples)

        # Update speed data for display and CSV export
        self.update_speed_data(samples)
        
        self.current_idx += samples
        self.listener_count += 1

        self.data_updated = True

    def update_speed_data(self, samples):
        """Update speed data for all enabled channels"""
        for i in range(self.num_channels):
            if not self.enabled_channels[i].get():
                continue

            start_idx = max(0, self.current_idx - samples)
            ch_data = self.data[i, start_idx:self.current_idx]

            self._speed_accumulator[i].extend(ch_data.tolist())

            # Process complete segments and save to CSV
            while len(self._speed_accumulator[i]) >= self.speed_down_sample_factor:
                seg = self._speed_accumulator[i][:self.speed_down_sample_factor]
                self._speed_accumulator[i] = self._speed_accumulator[i][self.speed_down_sample_factor:]

                t = np.arange(len(seg)) / self.input_rate

                speed, transition = self.compute_speed(np.array(seg), t, self.voltage_range[i])
                self.speed_data[i].append(speed)
                self.transition_counts[i] += transition

        # Check if all enabled channels have new speed data, then write to CSV
        active_indices = [i for i, enabled in enumerate(self.enabled_channels) if enabled.get()]
        min_speed_len = min([len(self.speed_data[i]) for i in active_indices] or [0])
        
        # Write new speed data points to CSV
        while self.csv_frame_index < min_speed_len:
            # Calculate time for this speed point
            speed_time = (self.csv_frame_index + 0.5) * self.speed_down_sample_factor / self.input_rate
            
            # Collect speed values for all active channels
            speed_values = [self.speed_data[i][self.csv_frame_index] for i in active_indices]
            
            # Write to CSV
            if self.csv_writer:
                self.csv_writer.writerow([f"{speed_time:.6f}"] + [f"{s:.6f}" for s in speed_values])
            
            self.csv_frame_index += 1

    def compute_speed(self, data, time, voltage_range):
        """Compute speed from raw voltage data"""
        is_transition = 0
        delta_voltage = voltage_range[1] - voltage_range[0]

        if abs(delta_voltage) < 1e-6:
            return 0.0, 0
        
        thresh = 3 / 5 * delta_voltage

        diff_data = np.diff(data)
        ind = np.where(np.abs(diff_data) > 3)[0]

        for i in ind:
            is_transition = 1
            if diff_data[i] < -thresh:
                data[i + 1:] += delta_voltage
            elif diff_data[i] > thresh:
                data[i + 1:] -= delta_voltage

        data_deg = (data / delta_voltage) * 360

        if len(data_deg) >= 20:
            delta_deg = np.mean(data_deg[-10:]) - np.mean(data_deg[:10])

            if delta_deg > 200: 
                delta_deg -= 360
            elif delta_deg < -200: 
                delta_deg += 360

            duration = np.mean(time[-10:]) - np.mean(time[:10])
            speed = delta_deg / duration if duration > 1e-6 else 0.0
        else:
            speed = 0.0
            
        return speed, is_transition

    def save_raw_data_to_ast2(self, samples):
        """Save raw voltage data to AST2 file immediately (no downsampling)"""
        if not self.file or self.file.closed:
            return
            
        try:
            # Get active channels
            active_indices = [i for i, enabled in enumerate(self.enabled_channels) if enabled.get()]
            
            # Get the raw voltage data that was just collected
            start_idx = self.current_idx
            end_idx = self.current_idx + samples
            
            interleaved = np.empty((samples, len(active_indices)), dtype=np.int16)
            for col, ch in enumerate(active_indices):
                interleaved[:, col] = (self.data[ch, start_idx:end_idx] * self.scale).astype(np.int16)
            
            # Write to file immediately
            self.file.write(interleaved.ravel().tobytes())
            
        except Exception as e:
            print(f"Error saving raw voltage data to AST2: {e}")

    def stop_tracking(self):
        """Stop acquisition and close AST2 and CSV files"""
        if not self.running:
            return

        # Close AST2 file
        if hasattr(self, 'file') and self.file and not self.file.closed:
            self.file.close()
            self.file = None
            print(f"AST2 file saved: {self.ast2_file_path}")

        # Close CSV file
        if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"CSV file saved: {self.csv_file_path}")

        # Stop DAQ task
        if self.task:
            try:
                self.task.StopTask()
                self.task.ClearTask()
            except Exception as e:
                print(f"Error stopping task: {e}")
            finally:
                self.task = None

        self.running = False
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
            self._update_job = None

    def stop_update_loop(self):
        """Stop the update loop"""
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
            self._update_job = None
    
    def update_display_loop(self):
        """Update display periodically"""
        if not self.running:
            return
        if self.data_updated:
            self.update_display()
            self.data_updated = False
        self._update_job = self.root.after(200, self.update_display_loop)

    def update_display(self):
        """Update the plot display"""
        if self.current_idx < self.speed_down_sample_factor:
            return
            
        for i in range(self.num_channels):
            if not self.enabled_channels[i].get():
                continue
                
            view_speed = self.view_mode[i].get()
            if view_speed and self.speed_data[i]:
                y_data = np.array(self.speed_data[i])
                x_data = np.linspace(0, len(y_data) / (self.input_rate / self.speed_down_sample_factor), len(y_data))
                self.ax[i].set_ylabel("Speed (°/s)")
            else:
                end_idx = min(self.current_idx, len(self.time))
                y_data = self.data[i, :end_idx:self.down_sample]
                x_data = self.time[:end_idx:self.down_sample]
                y_data = y_data[:len(x_data)]
                self.ax[i].set_ylabel("Voltage (V)")
                
            self.lines[i].set_data(x_data, y_data)
            
            if len(x_data) > 0:
                self.ax[i].set_xlim(0, x_data[-1])
                
            if self.auto_scale[i].get():
                self.ax[i].set_autoscale_on(True)
                self.ax[i].relim()
                self.ax[i].autoscale_view()
            else:
                try:
                    xlim_str = self.x_range[i].get().strip()
                    ylim_str = self.y_range[i].get().strip()
                    
                    if xlim_str:
                        xlim = list(map(float, xlim_str.split(',')))
                        if len(xlim) == 2:
                            self.ax[i].set_xlim(xlim)
                            
                    if ylim_str:
                        ylim = list(map(float, ylim_str.split(',')))
                        if len(ylim) == 2:
                            self.ax[i].set_ylim(ylim)
                except ValueError:
                    pass
        
        self.canvas.draw_idle()
        self._update_status_bar()
        
    def _update_status_bar(self):
        """Update status bar with current speed information"""
        avg_speeds = []
        for i in range(self.num_channels):
            if self.speed_data[i]:
                recent_speeds = self.speed_data[i][-5:] if len(self.speed_data[i]) >= 5 else self.speed_data[i]
                avg_speed = np.mean(recent_speeds)
            else:
                avg_speed = 0
            avg_speeds.append(avg_speed)
            
        transitions = self.transition_counts.astype(int)
        
        status = " | ".join([
            f"Ch{i}: {avg_speeds[i]:.1f}°/s, {transitions[i]} trans." 
            for i in range(self.num_channels) if self.enabled_channels[i].get()
        ])
        self.status_var.set(status)

    def apply_axis_limits(self):
        """Apply axis limit changes"""
        self.data_updated = True

    def auto_calibrate_voltage_range(self):
        """Auto-calibrate voltage range from signal transitions"""
        if self.current_idx == 0:
            messagebox.showinfo("Info", "No data available for calibration.")
            return
            
        for i in range(self.num_channels):
            if self.enabled_channels[i].get():
                signal = self.data[i, :self.current_idx]
                
                diff = np.diff(signal)
                jump_indices = np.where(np.abs(diff) > 2.5)[0]
                
                if len(jump_indices) > 0:
                    v_min = np.min(signal)
                    v_max = np.max(signal)
                    
                    v_margin = (v_max - v_min) * 0.1
                    self.voltage_range[i] = [v_min - v_margin, v_max + v_margin]
                else:
                    messagebox.showinfo("Info", f"No significant transitions found in Channel {i}.")
                    
        messagebox.showinfo("Done", "Voltage range calibrated from signal.")

class ExperimentGUI:
    """Main GUI application for fear training experiments"""
    def __init__(self, root, arduino_port="COM9", daq_device="Dev2"):
        # Initialize GUI components and variables
        self.root = root
        self.root.title("Experiment Setting")
        self.root.state('zoomed')

        self.arduino_port = arduino_port
        self.daq_device = daq_device

        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            print("Arduino port opened.")
            time.sleep(2)
        except serial.SerialException as e:
            messagebox.showerror("Arduino Error", f"Failed to open {self.arduino_port}: {e}, if you don't need to control fiber photometry, please ignore it!")
            self.arduino = None

        # Initialize data structures
        self.cameras = []
        self.camera_views = []
        self.timeline = []
        self.save_dir = os.getcwd()
        self.video_start_time = None
        self.timber_start = None
        self.event_log = []  # Log for experimental events
        self.timestamp_log = []  # Log for device timestamps
        self.opto_genetic_log = []  # Log for optogenetic stimulations
        
        # New state variables
        self.is_running = False
        self.timeline_mode = tk.BooleanVar(value=True)  # Timeline mode on by default

        # GUI variables
        self.exposure_time = tk.DoubleVar(value=4000.0)
        self.resolution_choice = tk.StringVar(value="1280x1024")
        self.record_rate = tk.DoubleVar(value=90.0)
        self.sample_rate = tk.DoubleVar(value=90.0)
        self.pixel_type = PixelType_Mono8
        self.selected_audio_device = None
        self.user_filename_prefix = tk.StringVar(value="experiment")
        self.enable_speed_sensor = tk.BooleanVar(value=False)
        self.toggle_wait_optogenetics = tk.BooleanVar(value=False)
        self.toggle_sound_optogenetics = tk.BooleanVar(value=False)
        self.toggle_sound_shock_optogenetics = tk.BooleanVar(value=False)
        self.stimulation_locations = []
        self.speed_encoder_gui = None

        # Optogenetics parameters
        self.opto_frequency = tk.DoubleVar(value=20.0)  # Hz
        self.opto_pulse_width = tk.DoubleVar(value=10.0)  # ms
        self.opto_duration = tk.DoubleVar(value=1000.0)  # ms
        self.opto_delay = tk.DoubleVar(value=0.0)  # s
        self.stimulation_settings = []

        # Setup UI components
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        # Add Timeline mode toggle at the top
        self.timeline_mode_frame = ttk.LabelFrame(left_panel, text="Timeline Mode")
        self.timeline_mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(self.timeline_mode_frame, text="Timeline On", variable=self.timeline_mode,
                       command=self.toggle_timeline_mode).grid(row=0, column=0, padx=5, pady=5)

        self.speed_frame = ttk.LabelFrame(left_panel, text="Speed Sensor Control")
        self.speed_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(self.speed_frame, text="Enable Speed Sensor", variable=self.enable_speed_sensor,
                       command=self.toggle_speed_sensor).grid(row=0, column=0, padx=5, pady=5)

        self.speed_settings_frame = ttk.Frame(self.speed_frame)
        self.speed_settings_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.speed_settings_frame, text="Input Rate (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.speed_input_rate = tk.IntVar(value=1000)
        ttk.Entry(self.speed_settings_frame, textvariable=self.speed_input_rate, width=10).grid(row=0, column=1, padx=5, pady=5)

        self.toggle_speed_sensor()

        self.config_frame = ttk.LabelFrame(left_panel, text="Block Configuration")
        self.config_frame.pack(fill=tk.X, padx=5, pady=5)

        opto_frame = ttk.Frame(self.config_frame) 
        opto_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        wait_opto_frame = ttk.Frame(opto_frame)
        wait_opto_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(wait_opto_frame, text="Show Wait+Opto", variable=self.toggle_wait_optogenetics,
                        command=self.update_block_types).pack(side=tk.LEFT, padx=5)

        sound_opto_frame = ttk.Frame(opto_frame)
        sound_opto_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(sound_opto_frame, text="Show Sound+Opto", variable=self.toggle_sound_optogenetics,
                        command=self.update_block_types).pack(side=tk.LEFT, padx=5)

        sound_shock_opto_frame = ttk.Frame(opto_frame)
        sound_shock_opto_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(sound_shock_opto_frame, text="Show Sound+Shock+Opto", variable=self.toggle_sound_shock_optogenetics,
                        command=self.update_block_types).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.config_frame, text="Block Type:").grid(row=1, column=0, sticky="w", pady=2)
        self.block_type = tk.StringVar(value="Wait")
        self.type_combo = ttk.Combobox(self.config_frame, textvariable=self.block_type, width=15)
        self.type_combo.grid(row=1, column=1, sticky="ew", pady=2)
        self.type_combo.bind("<<ComboboxSelected>>", self.on_type_change)

        self.update_block_types()

        self.duration_frame = ttk.Frame(self.config_frame)
        self.duration_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        
        ttk.Label(self.duration_frame, text="Duration (s):").pack(side=tk.LEFT)
        self.block_duration = tk.StringVar(value="5")
        ttk.Entry(self.duration_frame, textvariable=self.block_duration, width=20).pack(side=tk.LEFT, padx=5)

        self.shock_frame = ttk.Frame(self.config_frame)
        
        ttk.Label(self.shock_frame, text="Sound Duration (s):").grid(row=0, column=0, sticky="w")
        self.sound_duration = tk.StringVar(value="5")
        ttk.Entry(self.shock_frame, textvariable=self.sound_duration, width=14).grid(row=0, column=1)
        
        ttk.Label(self.shock_frame, text="Shock Duration (s):").grid(row=1, column=0, sticky="w")
        self.shock_lead = tk.StringVar(value="2")
        ttk.Entry(self.shock_frame, textvariable=self.shock_lead, width=14).grid(row=1, column=1)

        self.stimulation_frame = ttk.LabelFrame(self.config_frame, text="Optogenetic Stimulation Settings")
        self.stimulation_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)

        stimulation_list_frame = ttk.Frame(self.stimulation_frame)
        stimulation_list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        stimulation_y_scrollbar = ttk.Scrollbar(stimulation_list_frame, orient="vertical")
        stimulation_y_scrollbar.pack(side="right", fill="y")

        stimulation_x_scrollbar = ttk.Scrollbar(stimulation_list_frame, orient="horizontal")
        stimulation_x_scrollbar.pack(side="bottom", fill="x")

        self.stimulation_listbox = tk.Listbox(
            stimulation_list_frame,
            height=3,
            yscrollcommand=stimulation_y_scrollbar.set,
            xscrollcommand=stimulation_x_scrollbar.set
        )
        self.stimulation_listbox.pack(side="left", fill="both", expand=True)

        stimulation_y_scrollbar.config(command=self.stimulation_listbox.yview)
        stimulation_x_scrollbar.config(command=self.stimulation_listbox.xview)

        # Optogenetic stimulation parameters
        stimulation_params_frame = ttk.LabelFrame(self.stimulation_frame, text="Stimulation Parameters")
        stimulation_params_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(stimulation_params_frame, text="Frequency (Hz):").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(stimulation_params_frame, textvariable=self.opto_frequency, width=10).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(stimulation_params_frame, text="Pulse Width (ms):").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(stimulation_params_frame, textvariable=self.opto_pulse_width, width=10).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(stimulation_params_frame, text="Duration (ms):").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(stimulation_params_frame, textvariable=self.opto_duration, width=10).grid(row=2, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(stimulation_params_frame, text="Delay (s):").grid(row=3, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(stimulation_params_frame, textvariable=self.opto_delay, width=10).grid(row=3, column=1, padx=5, pady=2, sticky="w")

        stimulation_control_frame = ttk.Frame(self.stimulation_frame)
        stimulation_control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(stimulation_control_frame, text="Add Sti", command=self.add_stimulation).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(stimulation_control_frame, text="Delete Sti", command=self.delete_stimulation).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(stimulation_control_frame, text="Test Sti", command=self.test_stimulation).grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.stimulation_frame.grid_remove()

        btn_frame = ttk.Frame(self.config_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        row1 = ttk.Frame(btn_frame)
        row1.pack(side=tk.TOP, pady=2)
        ttk.Button(row1, text="Add Block", command=self.add_block, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Remove Block", command=self.remove_selected_block, width=12).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(btn_frame)
        row2.pack(side=tk.TOP, pady=2)

        ttk.Button(row2, text="Clear All", command=self.clear_blocks, width=26).pack(side=tk.TOP)

        self.file_frame = ttk.LabelFrame(left_panel, text="File Settings")
        self.file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.file_frame, text="Filename Prefix:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(self.file_frame, textvariable=self.user_filename_prefix, width=17).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(self.file_frame, text="Choose Save Directory", command=self.choose_directory, width=30).grid(row=1, column=0, columnspan=2, pady=5)

        self.device_frame = ttk.LabelFrame(left_panel, text="Device Control")
        self.device_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(self.device_frame, text="Add Camera", command=self.add_camera, width=14).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.device_frame, text="Remove Camera", command=self.remove_camera, width=14).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.device_frame, text="Edit Camera", command=self.edit_camera, width=14).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.device_frame, text="Clear All Cameras", command=self.clear_all_cameras, width=14).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.device_frame, text="Select Speaker", command=self.select_devices, width=14).grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(self.device_frame, text="Test Speaker", command=self.test_speaker, width=14).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(self.device_frame, text="Start Preview All", command=self.start_preview_all, width=14).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.device_frame, text="Stop Preview All", command=self.stop_preview_all, width=14).grid(row=2, column=1, padx=5, pady=5)
        
        # Modified START/STOP button
        self.start_stop_button = ttk.Button(self.device_frame, text="START", command=self.toggle_start_stop, style="Accent.TButton", width=14)
        self.start_stop_button.grid(row=4, column=0, padx=5, pady=5)
        
        ttk.Button(self.device_frame, text="EXIT", command=self.exit, style="Accent.TButton", width=14).grid(row=4, column=1, padx=5, pady=5)

        status_frame = ttk.Frame(left_panel)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.recording_label = ttk.Label(status_frame, text="Status: Ready", font=("Arial", 10))
        self.recording_label.pack(anchor='center')

        timber_frame = ttk.Frame(left_panel)
        timber_frame.pack(fill=tk.X, padx=10, pady=20)

        self.time_label = ttk.Label(timber_frame, text="00:00:00:000", font=("Arial", 12, "bold"), foreground="#FF0000")
        self.time_label.pack(anchor='center', pady=(5, 0))

        middle_panel = ttk.Frame(main_frame)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        middle_panel_top = ttk.Frame(middle_panel)
        middle_panel_top.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        middle_panel_top.configure(width=960) 
        middle_panel_top.configure(height=300) 

        timeline_frame = ttk.LabelFrame(middle_panel_top, text="Timeline Sequence")
        timeline_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        timeline_listbox_frame = ttk.Frame(timeline_frame)
        timeline_listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.timeline_listbox = tk.Listbox(timeline_listbox_frame, font=("Arial", 10), height=5)
        timeline_scrollbar_y = ttk.Scrollbar(timeline_listbox_frame, orient="vertical", command=self.timeline_listbox.yview)
        timeline_scrollbar_x = ttk.Scrollbar(timeline_listbox_frame, orient="horizontal", command=self.timeline_listbox.xview)

        self.timeline_listbox.configure(yscrollcommand=timeline_scrollbar_y.set, xscrollcommand=timeline_scrollbar_x.set)

        self.timeline_listbox.grid(row=0, column=0, sticky="nsew")
        timeline_scrollbar_y.grid(row=0, column=1, sticky="ns")
        timeline_scrollbar_x.grid(row=1, column=0, sticky="ew")

        timeline_listbox_frame.grid_rowconfigure(0, weight=1)
        timeline_listbox_frame.grid_columnconfigure(0, weight=1)

        self.timeline_listbox.bind("<<ListboxSelect>>", self.sync_selection)

        camera_mgmt_frame = ttk.LabelFrame(middle_panel_top, text="Camera Management")
        camera_mgmt_frame.grid(row=0, column=1, sticky="nsew")

        camera_listbox_frame = ttk.Frame(camera_mgmt_frame)
        camera_listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.camera_listbox = tk.Listbox(camera_listbox_frame, height=4)
        camera_scrollbar_y = ttk.Scrollbar(camera_listbox_frame, orient="vertical", command=self.camera_listbox.yview)
        camera_scrollbar_x = ttk.Scrollbar(camera_listbox_frame, orient="horizontal", command=self.camera_listbox.xview)

        self.camera_listbox.configure(yscrollcommand=camera_scrollbar_y.set, xscrollcommand=camera_scrollbar_x.set)

        self.camera_listbox.grid(row=0, column=0, sticky="nsew")
        camera_scrollbar_y.grid(row=0, column=1, sticky="ns")
        camera_scrollbar_x.grid(row=1, column=0, sticky="ew")

        camera_listbox_frame.grid_rowconfigure(0, weight=1)
        camera_listbox_frame.grid_columnconfigure(0, weight=1)

        middle_panel_top.grid_columnconfigure(0, weight=1)
        middle_panel_top.grid_columnconfigure(1, weight=1)
        middle_panel_top.grid_rowconfigure(0, weight=1)

        self.video_container = ttk.LabelFrame(middle_panel, text="Camera Previews")
        self.video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(1, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(1, weight=1)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.speed_encoder_gui = SpeedEncoderGUI(self.root, right_panel, setup_dialog.daq_device)

        self.root.style = ttk.Style()
        self.root.style.configure("Accent.TButton", font=("Arial", 10, "bold"), foreground="black", background="#4CAF50")
        self.root.style.configure("Stop.TButton", font=("Arial", 10, "bold"), foreground="black", background="#FF0000")
        
        # Initial state update
        self.toggle_timeline_mode()
        
    def toggle_timeline_mode(self):
        """Enable/disable timeline configuration based on mode"""
        state = tk.NORMAL if self.timeline_mode.get() else tk.DISABLED
        
        # Enable/disable all widgets in the config frame
        for child in self.config_frame.winfo_children():
            try:
                child.configure(state=state)
            except:
                # Some widgets might not have state configuration
                pass
                
        # Also enable/disable the timeline listbox
        self.timeline_listbox.configure(state=state)
        
    def toggle_start_stop(self):
        """Toggle between start and stop states"""
        if not self.is_running:
            self.start_experiment()
        else:
            self.stop_experiment()
            
    def start_experiment(self):
        """Start the experiment based on current mode"""
        # Change button to STOP state
        self.is_running = True
        self.start_stop_button.configure(text="STOP", style="Stop.TButton")
        
        # Disable other controls while running
        self.set_controls_state(tk.DISABLED)
        
        # Start experiment based on mode
        if self.timeline_mode.get():
            if not self.timeline:
                messagebox.showerror("Error", "Timeline is empty")
                self.stop_experiment()
                return
            threading.Thread(target=self._run_timeline, daemon=True).start()
        else:
            threading.Thread(target=self._run_free_mode, daemon=True).start()
            
    def stop_experiment(self):
        """Stop the experiment and all data collection"""
        self.is_running = False
        
        # Re-enable controls
        self.set_controls_state(tk.NORMAL)
        
        # Change button back to START state
        self.start_stop_button.configure(text="START", style="Accent.TButton")
        
    def set_controls_state(self, state):
        """Enable/disable controls based on state"""
        # Enable/disable timeline mode toggle
        for child in self.timeline_mode_frame.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass
                
        # Enable/disable device controls
        for child in self.device_frame.winfo_children():
            try:
                if child != self.start_stop_button:  # Don't disable the start/stop button
                    child.configure(state=state)
            except:
                pass
                
        # Enable/disable speed sensor controls
        for child in self.speed_frame.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass
                
        # Enable/disable file settings
        for child in self.file_frame.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass
        
    def _run_free_mode(self):
        """Run experiment in free mode (no timeline)"""
        self.recording_label.config(text="Status: Running Experiment (Free Mode)")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = self.user_filename_prefix.get()

        # Start speed sensor if enabled
        speed_sensor_enabled = False
        speed_start = 0
        if self.enable_speed_sensor.get() and self.speed_encoder_gui:
            try:
                self.speed_encoder_gui.input_rate = self.speed_input_rate.get()
                speed_file_base = f"{prefix}_speed_{timestamp}"
                speed_file_path = os.path.join(self.save_dir, speed_file_base)
                if self.speed_encoder_gui.start_tracking(speed_file_path):
                    speed_sensor_enabled = True
                    speed_start = time.time()
                    self.timestamp_log.append(("Speed Sensor", "Start", speed_start))
            except Exception as e:
                messagebox.showerror("Speed Sensor Error", f"Failed to start speed sensor: {e}")

        if self.cameras:
            # Start camera recording
            camera_timestamps = []
            for camera in self.cameras:
                filename = f"{prefix}_{timestamp}_cam{camera.camera_id}.avi"
                save_path = os.path.join(self.save_dir, filename)
                try:
                    start_time = camera.start_record(save_path)
                    camera_timestamps.append((camera.camera_id, start_time))
                    self.timestamp_log.append((f"Camera {camera.camera_id}", "Start", start_time))
                except Exception as e:
                    messagebox.showerror("Recording Error", f"Failed to start recording for camera {camera.camera_id}: {e}")
            
            self.video_start_time  = time.time()
            self.timber_start = self.video_start_time
        else:
            self.timber_start = time.time()
            print(self.timber_start)

        self.update_timer()
        experiment_start = time.time()
        self.timestamp_log.append(("Experiment", "Start", experiment_start))

        # Wait until stop is requested
        while self.is_running:
            time.sleep(0.1)  # Small sleep to prevent CPU overuse
        
        if self.cameras:
            for camera in self.cameras:
                if camera.recording:
                    end_time = camera.stop_record()
                    self.timestamp_log.append((f"Camera {camera.camera_id}", "End", end_time))

        if speed_sensor_enabled and self.speed_encoder_gui:
            self.speed_encoder_gui.stop_tracking()
            speed_end = time.time()
            self.timestamp_log.append(("Speed Sensor", "End", speed_end))
            print("Speed sensor recording stopped")
        
        # Record end times
        experiment_end = time.time()
        self.timestamp_log.append(("Experiment", "End", experiment_end))

        self.save_logs(prefix, timestamp)
        # clear previous logs
        self.timestamp_log = []
        self.recording_label.config(text="Status: Experiment Complete")

    def toggle_speed_sensor(self):
        state = tk.NORMAL if self.enable_speed_sensor.get() else tk.DISABLED
        for child in self.speed_settings_frame.winfo_children():
            child.configure(state=state)

        if self.enable_speed_sensor.get():
            if self.speed_encoder_gui:
                self.speed_encoder_gui.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                self.speed_encoder_gui.ch_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                self.speed_encoder_gui.btn_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                self.speed_encoder_gui.status_label.pack(fill=tk.X, padx=5, pady=5)
        else:
            if self.speed_encoder_gui:
                self.speed_encoder_gui.plot_frame.pack_forget()
                self.speed_encoder_gui.ch_frame.pack_forget()
                self.speed_encoder_gui.btn_frame.pack_forget()
                self.speed_encoder_gui.status_label.pack_forget()

    def update_block_types(self):
        block_types = ["Wait", "Sound"]
        
        if self.toggle_wait_optogenetics.get():
            block_types.append("Wait+Optogenetics")
            
        if self.toggle_sound_optogenetics.get():
            block_types.append("Sound+Optogenetics")
            
        if self.toggle_sound_shock_optogenetics.get():
            block_types.append("Sound+Shock+Optogenetics")
            
        if "Sound+Shock" not in block_types:
            block_types.append("Sound+Shock")
            
        self.type_combo['values'] = block_types
        
        if self.block_type.get() not in block_types:
            self.block_type.set(block_types[0])
            
        self.on_type_change()

    def on_type_change(self, event=None):
        if not hasattr(self, 'stimulation_frame'):
            return
    
        btype = self.block_type.get()
        
        if "Optogenetics" in btype:
            self.stimulation_frame.grid()
            self.stimulation_locations = []
            self.stimulation_listbox.delete(0, tk.END)
        else:
            self.stimulation_frame.grid_remove()
            
        if btype == "Sound+Shock" or btype == "Sound+Shock+Optogenetics":
            self.duration_frame.grid_remove()
            self.shock_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        else:
            self.shock_frame.grid_remove()
            self.duration_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)

    def add_stimulation(self):
        btype = self.block_type.get()
        
        if btype in ["Sound+Shock", "Sound+Shock+Optogenetics"]:
            try:
                max_duration = float(self.sound_duration.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid sound duration")
                return
        elif btype in ["Wait", "Sound", "Wait+Optogenetics", "Sound+Optogenetics"]:
            try:
                max_duration = float(self.block_duration.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid duration")
                return
        else:
            max_duration = 30.0
        
        try:
            frequency = self.opto_frequency.get()
            pulse_width = self.opto_pulse_width.get()
            duration = self.opto_duration.get()
            delay = self.opto_delay.get()
            
            if frequency <= 0 or frequency > 500:
                messagebox.showerror("Error", "Frequency must be between 0 and 500 Hz")
                return
            
            max_pulse_width = (1000.0 / frequency)  # ms
            if pulse_width <= 0 or pulse_width > max_pulse_width:
                messagebox.showerror("Error", f"Pulse width must be between 0 and {max_pulse_width:.2f} ms (1/frequency)")
                return
            
            if duration <= 0:
                messagebox.showerror("Error", "Duration must be positive")
                return
            
            if delay < 0 or delay > max_duration:
                messagebox.showerror("Error", f"Delay must be between 0 and {max_duration:.1f} seconds")
                return
            
            stim_end_time = delay + (duration / 1000.0)
            if stim_end_time > max_duration:
                messagebox.showerror("Error", f"Stimulation (delay + duration) exceeds block duration ({max_duration:.1f}s)")
                return
            
            stim_setting = {
                'frequency': frequency,
                'pulse_width': pulse_width,
                'duration': duration,
                'delay': delay
            }
            self.stimulation_settings.append(stim_setting)
            self.update_stimulation_list()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values for all parameters")
            return

    def delete_stimulation(self):
        selection = self.stimulation_listbox.curselection()
        if selection:
            index = selection[0]
            del self.stimulation_settings[index]
            self.update_stimulation_list()

    def update_stimulation_list(self):
        self.stimulation_listbox.delete(0, tk.END)
        for i, setting in enumerate(self.stimulation_settings):
            display_text = (f"#{i+1}: {setting['frequency']:.1f}Hz, "
                        f"{setting['pulse_width']:.1f}ms pulse, "
                        f"{setting['duration']:.1f}ms duration, "
                        f"{setting['delay']:.1f}s delay")
            self.stimulation_listbox.insert(tk.END, display_text)

    def test_stimulation(self):
        try:
            frequency = self.opto_frequency.get()
            pulse_width = self.opto_pulse_width.get()
            duration = self.opto_duration.get()
            
            if frequency <= 0 or frequency > 500:
                messagebox.showerror("Error", "Frequency must be between 0 and 500 Hz")
                return
            
            max_pulse_width = (1000.0 / frequency)
            if pulse_width <= 0 or pulse_width > max_pulse_width:
                messagebox.showerror("Error", f"Pulse width must be between 0 and {max_pulse_width:.2f} ms")
                return
            
            if duration <= 0:
                messagebox.showerror("Error", "Duration must be positive")
                return
            
            self.send_opto_stimulation(frequency, pulse_width, duration, 0)
            messagebox.showinfo("Test", "Test stimulation sent to Arduino")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send test stimulation: {e}")
            
    def send_opto_stimulation(self, frequency, pulse_width, duration, delay):
        try:
            if self.arduino and self.arduino.is_open:
                if delay > 0:
                    time.sleep(delay)
                
                pulse_width_us = int(pulse_width * 1000)
                command = f"O,{int(frequency)},{pulse_width_us},{int(duration)}\n"
                
                self.arduino.write(command.encode())
                print(f"Optogenetic stimulation sent: {frequency}Hz, {pulse_width}ms pulse, {duration}ms duration, {delay}s delay")
            else:
                print("Arduino not connected, cannot send stimulation")
        except Exception as e:
            print(f"Error sending optogenetic stimulation: {e}")
            
    def test_speaker(self):
        self.play_sound(1.0)

    def add_block(self):
        btype = self.block_type.get()
        
        if "Optogenetics" in btype and not self.stimulation_settings:
            messagebox.showerror("Error", "Please add at least one stimulation setting for optogenetics blocks")
            return
            
        if btype == "Sound+Shock" or btype == "Sound+Shock+Optogenetics":
            try:
                sound_dur = float(self.sound_duration.get())
                shock_lead = float(self.shock_lead.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for Sound+Shock parameters")
                return
                
            if sound_dur <= 0 or shock_lead < 0:
                messagebox.showerror("Error", "Sound duration and shock duration must be positive")
                return
                
            if shock_lead > sound_dur:
                messagebox.showerror("Error", "Shock start time must be less than or equal to sound duration")
                return
                
            if "Optogenetics" in btype:
                self.timeline.append((btype, sound_dur, shock_lead, self.stimulation_settings.copy()))
                self.timeline_listbox.insert(tk.END, 
                    f"{btype} - Sound {sound_dur:.1f}s, Shock start {shock_lead:.1f}s before end, "
                    f"Stimulations: {len(self.stimulation_settings)}")
            else:
                self.timeline.append((btype, sound_dur, shock_lead))
                self.timeline_listbox.insert(tk.END, 
                    f"{btype} - Sound {sound_dur:.1f}s, Shock start {shock_lead:.1f}s before end")
                    
        elif btype == "Wait+Optogenetics":
            try:
                dur = float(self.block_duration.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid duration")
                return
                
            self.timeline.append((btype, dur, self.stimulation_settings.copy()))
            self.timeline_listbox.insert(tk.END, 
                f"{btype} - {dur:.1f}s, Stimulations: {len(self.stimulation_settings)}")
                
        elif btype == "Sound+Optogenetics":
            try:
                dur = float(self.block_duration.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid duration")
                return
                
            self.timeline.append((btype, dur, self.stimulation_settings.copy()))
            self.timeline_listbox.insert(tk.END, 
                f"{btype} - {dur:.1f}s, Stimulations: {len(self.stimulation_settings)}")
                
        else:  # Wait or Sound
            try:
                dur = float(self.block_duration.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid duration")
                return
                
            self.timeline.append((btype, dur))
            self.timeline_listbox.insert(tk.END, f"{btype} - {dur:.1f}s")
        
        self.stimulation_settings = []
        self.stimulation_listbox.delete(0, tk.END)

    def remove_selected_block(self):
        sel = self.timeline_listbox.curselection()
        if sel:
            index = sel[0]
            del self.timeline[index]
            self.timeline_listbox.delete(index)

    def clear_blocks(self):
        self.timeline.clear()
        self.timeline_listbox.delete(0, tk.END)

    def sync_selection(self, event):
        sel = self.timeline_listbox.curselection()
        if sel:
            index = sel[0]
            entry = self.timeline[index]
            btype = entry[0]
            self.block_type.set(btype)
            self.on_type_change()
            
            if "Optogenetics" in btype:
                if btype == "Wait+Optogenetics" or btype == "Sound+Optogenetics":
                    _, dur, settings = entry
                    self.block_duration.set(str(dur))
                    self.stimulation_settings = [s.copy() for s in settings]
                elif btype == "Sound+Shock+Optogenetics":
                    _, sound_dur, shock_lead, settings = entry
                    self.sound_duration.set(str(sound_dur))
                    self.shock_lead.set(str(shock_lead))
                    self.stimulation_settings = [s.copy() for s in settings]
                    
                self.update_stimulation_list()
            elif btype == "Sound+Shock":
                _, sound_dur, shock_lead = entry
                self.sound_duration.set(str(sound_dur))
                self.shock_lead.set(str(shock_lead))
            else:
                dur = entry[1]
                self.block_duration.set(str(dur))

    def choose_directory(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.save_dir = selected_dir

    def select_devices(self):
        mic_list = list_audio_devices()

        def apply_selection():
            self.selected_audio_device = mic_combo.get()
            sel_win.destroy()

        sel_win = tk.Toplevel(self.root)
        sel_win.title("Select Speaker")

        tk.Label(sel_win, text="Select Speaker:").pack()
        mic_combo = ttk.Combobox(sel_win, values=mic_list, width=40)
        mic_combo.pack(padx=10, pady=5)
        if mic_list:
            mic_combo.current(0)

        ttk.Button(sel_win, text="OK", command=apply_selection).pack(pady=10)

    def add_camera(self):
        popup = tk.Toplevel(self.root)
        popup.title("Add Camera")
        popup.resizable(False, False)
        popup.grab_set()
        
        ttk.Label(popup, text="Camera Type:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        camera_type = tk.StringVar(value="HKCamera")
        camera_type_combo = ttk.Combobox(popup, textvariable=camera_type, 
                                       values=["HKCamera", "OpenCVCamera", "FLIRCamera"], width=15)
        camera_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        device_select_frame = ttk.LabelFrame(popup, text="Device Selection")
        device_select_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        ttk.Label(device_select_frame, text="Select Device:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        device_var = tk.StringVar()
        device_combo = ttk.Combobox(device_select_frame, textvariable=device_var, width=20, state="readonly")
        device_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        hk_frame = ttk.LabelFrame(popup, text="HIK Camera Parameters")
        hk_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        opencv_frame = ttk.LabelFrame(popup, text="OpenCV Camera Parameters")
        opencv_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        flir_frame = ttk.LabelFrame(popup, text="FLIR Camera Parameters")
        flir_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        common_frame = ttk.Frame(popup)
        common_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(hk_frame, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        exposure_var = tk.DoubleVar(value=self.exposure_time.get())
        ttk.Entry(hk_frame, textvariable=exposure_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(hk_frame, text="Acquisition FPS:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        sample_rate_var = tk.DoubleVar(value=self.sample_rate.get())
        ttk.Entry(hk_frame, textvariable=sample_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(flir_frame, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        flir_exposure_var = tk.DoubleVar(value=self.exposure_time.get())
        ttk.Entry(flir_frame, textvariable=flir_exposure_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(flir_frame, text="Acquisition FPS:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        flir_sample_rate_var = tk.DoubleVar(value=self.sample_rate.get())
        ttk.Entry(flir_frame, textvariable=flir_sample_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(common_frame, text="Resolution:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        resolution_var = tk.StringVar(value=self.resolution_choice.get())
        resolution_menu = ttk.OptionMenu(common_frame, resolution_var, resolution_var.get(), 
                                       "1920x1080","1280x1024", "1280x720", "640x480")
        resolution_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(common_frame, text="Recording FPS:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        record_rate_var = tk.DoubleVar(value=self.record_rate.get())
        ttk.Entry(common_frame, textvariable=record_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        def update_device_list():
            ctype = camera_type.get()
            device_combo.set('')
            if ctype == "HKCamera":
                devices = HKCamera.list_devices()
                device_combo['values'] = devices
                if devices:
                    device_combo.current(0)
            elif ctype == "FLIRCamera":
                devices = FLIRCamera.list_devices()
                device_combo['values'] = devices
                if devices:
                    device_combo.current(0)
            else:  # OpenCVCamera
                devices = list_available_cameras()
                device_combo['values'] = devices
                if devices:
                    device_combo.current(0)

        def toggle_parameters():
            ctype = camera_type.get()
            update_device_list()
            
            if ctype == "HKCamera":
                hk_frame.grid()
                opencv_frame.grid_remove()
                flir_frame.grid_remove()
                device_select_frame.grid()
            elif ctype == "OpenCVCamera":
                hk_frame.grid_remove()
                opencv_frame.grid()
                flir_frame.grid_remove()
                device_select_frame.grid()
            else:  # FLIRCamera
                hk_frame.grid_remove()
                opencv_frame.grid_remove()
                flir_frame.grid()
                device_select_frame.grid()
        
        camera_type_combo.bind("<<ComboboxSelected>>", lambda e: toggle_parameters())
        update_device_list()
        toggle_parameters()

        def create_camera():
            try:
                camera_id = len(self.cameras)
                res_text = resolution_var.get()
                    
                width, height = map(int, res_text.split("x"))
                
                device_index = device_combo.current()
                
                if camera_type.get() == "FLIRCamera":
                    if device_index < 0:
                        messagebox.showerror("Error", "Please select a FLIR camera device")
                        return
                    camera = FLIRCamera(
                        self.root, 
                        self.arduino, 
                        camera_id=camera_id,
                        device_index=device_index
                    )
                    camera.set_parameters(
                        width=width,
                        height=height,
                        exposure_time=flir_exposure_var.get(),
                        sample_rate=flir_sample_rate_var.get(),
                        record_rate=record_rate_var.get()
                    )
                    camera.open()
                elif camera_type.get() == "HKCamera":
                    if device_index < 0:
                        messagebox.showerror("Error", "Please select a HIK camera device")
                        return
                    camera = HKCamera(
                        self.root, 
                        self.arduino, 
                        camera_id=camera_id,
                        device_index=device_index
                    )
                    camera.set_parameters(
                        width=width,
                        height=height,
                        exposure_time=exposure_var.get(),
                        sample_rate=sample_rate_var.get(),
                        record_rate=record_rate_var.get()
                    )
                    camera.open()
                else:
                    if device_index < 0:
                        messagebox.showerror("Error", "Please select an OpenCV camera device")
                        return
                    selected_camera_name = device_var.get()
                    camera_list = list_available_cameras()
                    if selected_camera_name not in camera_list:
                        messagebox.showerror("Error", "Selected camera device not found")
                        return
                    camera_index = camera_list.index(selected_camera_name)

                    camera = OpenCVCamera(self.root, self.arduino, 
                             camera_id=camera_id,
                             device_index=camera_index,
                             num_channels=1)
                    camera.set_parameters(
                        width=width,
                        height=height,
                        fps=record_rate_var.get()
                    )

                    camera.open()
                
                self.cameras.append(camera)
                self.create_camera_view(camera_id)
                self.update_camera_list()
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Camera Error", f"Failed to add camera: {e}")

        btn_frame = ttk.Frame(popup)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Add Camera", command=create_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=popup.destroy).pack(side=tk.LEFT, padx=5)

    def remove_camera(self):
        sel = self.camera_listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Please select a camera to remove")
            return
            
        index = sel[0]
        camera = self.cameras[index]
        if camera.previewing is True:
            camera.stop_preview()
        if camera.recording is True:
            camera.stop_record()
        camera.close()

        del self.cameras[index]

        self.update_camera_list()
        self.update_camera_views()
        messagebox.showinfo("Info", f"Camera {index} removed successfully")

    def edit_camera(self):
        sel = self.camera_listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Please select a camera to edit")
            return
            
        index = sel[0]
        camera = self.cameras[index]

        popup = tk.Toplevel(self.root)
        popup.title(f"Edit Camera {index}")
        popup.resizable(False, False)
        popup.grab_set()
        
        if isinstance(camera, FLIRCamera):
            ttk.Label(popup, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5)
            exposure_var = tk.DoubleVar(value=camera.exposure_time)
            ttk.Entry(popup, textvariable=exposure_var, width=10).grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(popup, text="Acquisition FPS:").grid(row=1, column=0, padx=5, pady=5)
            sample_rate_var = tk.DoubleVar(value=camera.sample_rate)
            ttk.Entry(popup, textvariable=sample_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(popup, text="Recording FPS:").grid(row=2, column=0, padx=5, pady=5)
            record_rate_var = tk.DoubleVar(value=camera.record_rate)
            ttk.Entry(popup, textvariable=record_rate_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        elif isinstance(camera, HKCamera):
            ttk.Label(popup, text="Exposure Time (us):").grid(row=0, column=0, padx=5, pady=5)
            exposure_var = tk.DoubleVar(value=camera.exposure_time)
            ttk.Entry(popup, textvariable=exposure_var, width=10).grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(popup, text="Acquisition FPS:").grid(row=1, column=0, padx=5, pady=5)
            sample_rate_var = tk.DoubleVar(value=camera.sample_rate)
            ttk.Entry(popup, textvariable=sample_rate_var, width=10).grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(popup, text="Recording FPS:").grid(row=2, column=0, padx=5, pady=5)
            record_rate_var = tk.DoubleVar(value=camera.record_rate)
            ttk.Entry(popup, textvariable=record_rate_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        else:
            ttk.Label(popup, text="Recording FPS:").grid(row=0, column=0, padx=5, pady=5)
            record_rate_var = tk.DoubleVar(value=camera.fps)
            ttk.Entry(popup, textvariable=record_rate_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        def apply_changes():
            if isinstance(camera, FLIRCamera):
                camera.exposure_time = exposure_var.get()
                camera.sample_rate  = sample_rate_var.get()
                camera.record_rate  = record_rate_var.get()
                try:
                    camera.cam.ExposureTime.SetValue(camera.exposure_time)
                    camera.cam.AcquisitionFrameRate.SetValue(camera.sample_rate)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to apply parameters: {e}")
                    
            elif isinstance(camera, HKCamera):
                camera.exposure_time = exposure_var.get()
                camera.sample_rate = sample_rate_var.get()
                camera.record_rate = record_rate_var.get()
                try:
                    camera.cam.MV_CC_SetFloatValue("ExposureTime", camera.exposure_time)
                    camera.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", camera.sample_rate)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to apply parameters: {e}")

            else:
                camera.fps = record_rate_var.get()
                if camera.capture and camera.capture.isOpened():
                    camera.capture.set(cv2.CAP_PROP_FPS, camera.fps)
            
            popup.destroy()
            messagebox.showinfo("Info", "Camera parameters updated")
        
        btn_frame = ttk.Frame(popup)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Apply", command=apply_changes).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=popup.destroy).grid(row=0, column=1, padx=5)

    def clear_all_cameras(self):
        if not self.cameras:
            messagebox.showinfo("Info", "No cameras to clear")
            return

        if not messagebox.askyesno("Confirm", "Are you sure you want to remove all cameras?"):
            return

        for camera in self.cameras:
            if camera.previewing is True:
                camera.stop_preview()
            if camera.recording is True:
                camera.stop_record()
            camera.close()

        self.cameras = []

        for view in self.camera_views:
            view["frame"].destroy()
        self.camera_views = []

        self.update_camera_list()

        messagebox.showinfo("Info", "All cameras have been removed")

    def update_camera_list(self):
        self.camera_listbox.delete(0, tk.END)
        for i, camera in enumerate(self.cameras):
            self.camera_listbox.insert(tk.END, f"Camera {i}: {camera.display_name}")

    def create_camera_view(self, camera_id):
        camera = self.cameras[camera_id]

        def update_callback(pil_img, cam_id):
            if cam_id == camera_id:
                self.update_image(pil_img, cam_id)
        
        camera.preview_callback = update_callback

        num_views = len(self.camera_views)
        row = num_views // 2
        col = num_views % 2

        camera_frame = ttk.LabelFrame(self.video_container, text=camera.display_name)
        camera_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        camera_frame.grid_rowconfigure(0, weight=1)
        camera_frame.grid_columnconfigure(0, weight=1)

        video_label = ttk.Label(camera_frame)
        video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.camera_views.append({
            "frame": camera_frame,
            "label": video_label,
            "camera_id": camera_id
        })

        self.video_container.grid_rowconfigure(row, weight=1)
        self.video_container.grid_columnconfigure(col, weight=1)

    def update_camera_views(self):
        for view in self.camera_views:
            view["frame"].destroy()
        self.camera_views = []

        for camera_id in range(len(self.cameras)):
            self.create_camera_view(camera_id)

    def update_image(self, pil_image, camera_id):
        if camera_id < len(self.camera_views):
            view = self.camera_views[camera_id]
            target_size = (480, 360)
            
            if pil_image.mode == 'L':
                pil_image = pil_image.convert('RGB')
                
            padded = ImageOps.pad(pil_image, target_size, color=0, centering=(0.5, 0.5))
            imgtk = ImageTk.PhotoImage(padded)
            
            view["label"].imgtk = imgtk
            view["label"].configure(image=imgtk)
            
            camera = self.cameras[camera_id]
            view["frame"].config(text=f"{camera.display_name} - FPS: {camera.current_fps:.1f}")
        else:
            print(f"[Warning]: The view corresponding to camera ID {camera_id} cannot be found")

    def start_preview_all(self):
        for camera in self.cameras:
            try:
                camera.start_preview(self.update_image)
            except Exception as e:
                messagebox.showerror("Preview Error", f"Failed to start preview: {str(e)}")
        self.recording_label.config(text="Status: Previewing All Cameras")

    def stop_preview_all(self):
        for camera in self.cameras:
            camera.stop_preview()
        self.recording_label.config(text="Status: Preview Stopped")

    def update_timer(self):
        if self.timber_start and (self.is_running or any(camera.recording for camera in self.cameras)):
            elapsed = time.time() - self.timber_start
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            millis = int(round((elapsed % 1) * 1000))
            self.time_label.config(text=f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}")
            self.root.after(50, self.update_timer)

    def run_timeline(self):
        if not self.timeline:
            messagebox.showerror("Error", "Timeline is empty")
            return
        if not self.cameras:
            messagebox.showerror("Error", "No cameras added")
            return
            
        threading.Thread(target=self._run_timeline, daemon=True).start()

    def _run_timeline(self):
        self.recording_label.config(text="Status: Running Experiment")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = self.user_filename_prefix.get()

        speed_sensor_enabled = False
        speed_start = speed_end = 0
        if self.enable_speed_sensor.get() and self.speed_encoder_gui:
            try:
                self.speed_encoder_gui.input_rate = self.speed_input_rate.get()
                
                # Generate base path (no extension needed, handled internally as AST2)
                speed_file_base = f"{prefix}_speed_{timestamp}"
                speed_file_path = os.path.join(self.save_dir, speed_file_base)
                
                # Start acquisition (converted to AST2 internally)
                if self.speed_encoder_gui.start_tracking(speed_file_path):
                    speed_sensor_enabled = True
                    speed_start = time.time()
                    print(f"Speed sensor started, AST2 file path: {self.speed_encoder_gui.ast2_file_path}")
                    self.timestamp_log.append(("Speed Sensor", "Start", speed_start))
            except Exception as e:
                messagebox.showerror("Speed Sensor Error", f"Failed to start speed sensor: {e}")

        if self.cameras:
            camera_timestamps = []
            for camera in self.cameras:
                filename = f"{prefix}_{timestamp}_cam{camera.camera_id}.avi"
                save_path = os.path.join(self.save_dir, filename)
                try:
                    start_time = camera.start_record(save_path)
                    camera_timestamps.append((camera.camera_id, start_time))
                    self.timestamp_log.append((f"Camera {camera.camera_id}", "Start", start_time))
                except Exception as e:
                    messagebox.showerror("Recording Error", f"Failed to start recording for camera {camera.camera_id}: {e}")
        
            self.video_start_time = time.time()
            self.timber_start = self.video_start_time
        else:
            self.timber_start = time.time()

        self.update_timer()

        experiment_start = time.time()
        self.timestamp_log.append(("Experiment", "Start", experiment_start))

        i = 0
        while i < len(self.timeline):
            entry = self.timeline[i]
            btype = entry[0]
            event_start = time.time()

            if btype == "Wait":
                dur = entry[1]
                time.sleep(dur)
                event_end = time.time()
                self.event_log.append((event_start, event_end, 0))
                
            elif btype == "Wait+Optogenetics":
                dur = entry[1]
                stim_settings = entry[2]
                
                for setting in stim_settings:
                    delay = setting['delay']
                    if delay < dur:
                        threading.Timer(delay, self.send_opto_stimulation, 
                                    args=(setting['frequency'], setting['pulse_width'], 
                                        setting['duration'], 0)).start()
                    self.opto_genetic_log.append((event_start, setting['frequency'], setting['pulse_width'], setting['duration'], setting['delay']))
                
                time.sleep(dur)
                event_end = time.time()
                self.event_log.append((event_start, event_end, 3))
                
            elif btype == "Sound":
                dur = entry[1]
                self.play_sound(dur)
                event_end = time.time()
                self.event_log.append((event_start, event_end, 1))
                
            elif btype == "Sound+Optogenetics":
                dur = entry[1]
                stim_settings = entry[2]
                
                for setting in stim_settings:
                    delay = setting['delay']
                    if delay < dur:
                        threading.Timer(delay, self.send_opto_stimulation,
                                    args=(setting['frequency'], setting['pulse_width'],
                                        setting['duration'], 0)).start()
                    self.opto_genetic_log.append((event_start, setting['frequency'], setting['pulse_width'], setting['duration'], setting['delay']))
                
                self.play_sound(dur)
                event_end = time.time()
                self.event_log.append((event_start, event_end, 4))
                
            elif btype == "Sound+Shock":
                sound_dur = entry[1]
                shock_lead = entry[2]
                shock_start = max(0, sound_dur - shock_lead)

                sample_rate = int(4 * 4000)
                t = np.arange(0, sound_dur, 1.0 / sample_rate)
                sound_wave = 0.5 * np.sin(2 * np.pi * 4000 * t)
                sd.play(sound_wave, samplerate=sample_rate, device=self.selected_audio_device, latency='low')

                if shock_start > 0:
                    time.sleep(shock_start)

                shock_event_start = time.time()
                self.trigger_stimulation(shock_lead)
                shock_event_end = time.time()

                sd.wait()

                event_end = time.time()
            
                self.event_log.append((event_start, event_end, 1))
                self.event_log.append((shock_event_start, shock_event_end, 2))
                
            elif btype == "Sound+Shock+Optogenetics":
                sound_dur = entry[1]
                shock_lead = entry[2]
                stim_settings = entry[3]
                shock_start = max(0, sound_dur - shock_lead)

                for setting in stim_settings:
                    delay = setting['delay']
                    if delay < sound_dur:
                        threading.Timer(delay, self.send_opto_stimulation,
                                    args=(setting['frequency'], setting['pulse_width'],
                                        setting['duration'], 0)).start()
                    self.opto_genetic_log.append((event_start, setting['frequency'], setting['pulse_width'], setting['duration'], setting['delay']))

                sample_rate = int(4 * 4000)
                t = np.arange(0, sound_dur, 1.0 / sample_rate)
                sound_wave = 0.5 * np.sin(2 * np.pi * 4000 * t)
                sd.play(sound_wave, samplerate=sample_rate, device=self.selected_audio_device, latency='low')

                if shock_start > 0:
                    time.sleep(shock_start)

                shock_event_start = time.time()
                self.trigger_stimulation(shock_lead)
                shock_event_end = time.time()

                sd.wait()

                event_end = time.time()

                self.event_log.append((event_start, event_end, 4))
                self.event_log.append((shock_event_start, shock_event_end, 5))
            
            i += 1

        if self.cameras:
            for camera in self.cameras:
                if camera.recording:
                    end_time = camera.stop_record()
                    self.timestamp_log.append((f"Camera {camera.camera_id}", "End", end_time))

        if speed_sensor_enabled and self.speed_encoder_gui:
            self.speed_encoder_gui.stop_tracking()
            speed_end = time.time()
            self.timestamp_log.append(("Speed Sensor", "End", speed_end))
            print("Speed sensor recording stopped")

        self.stop_experiment()
        experiment_end = time.time()
        self.timestamp_log.append(("Experiment", "End", experiment_end))

        self.save_logs(prefix, timestamp)
        # clear previous logs
        self.timestamp_log = []
        self.event_log = []
        self.opto_genetic_log = []

        self.recording_label.config(text="Status: Experiment Complete")
        print("Experiment complete")
        
    def save_logs(self, prefix, timestamp):
        if self.event_log:
            event_file = os.path.join(self.save_dir, f"{prefix}_{timestamp}_events.csv")
            try:
                with open(event_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["start_time", "end_time", "Event Type"])
                    for start, end, event in self.event_log:
                        writer.writerow([f"{start:.6f}", f"{end:.6f}", event])
                print(f"Event log saved: {event_file}")
            except Exception as e:
                print(f"Failed to save event log: {e}")
        else:
            print("No event log to save.")

        if self.timestamp_log:
            timestamp_file = os.path.join(self.save_dir, f"{prefix}_{timestamp}_timestamps.csv")
            try:
                with open(timestamp_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Device", "Action", "Timestamp"])
                    for device, action, timestamp in self.timestamp_log:
                        writer.writerow([device, action, f"{timestamp:.6f}"])
                print(f"Timestamp log saved: {timestamp_file}")
            except Exception as e:
                print(f"Failed to save timestamp log: {e}")
        else:
            print("No timestamp log to save.")

        if self.opto_genetic_log:
            optogenetics_file = os.path.join(self.save_dir, f"{prefix}_{timestamp}_optogenetics.csv")
            try:
                with open(optogenetics_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["start_time(for delay)", "frequency(Hz)", "pulse_width(ms)", "duration(ms)", "delay(s)"])
                    for start_time, freq, pulse_width, duration, delay in self.opto_genetic_log:
                        writer.writerow([f"{start_time:.6f}", freq, pulse_width, duration, delay])

                print(f"Optogenetics settings saved: {optogenetics_file}")
            except Exception as e:
                print(f"Failed to save optogenetics settings: {e}")
        else:
            print("No optogenetics settings to save.")

    def play_sound(self, duration, freq=4000, amp=0.5):
        sample_rate = 4 * freq
        t = np.arange(0, duration, 1 / sample_rate)
        sound_wave = amp * np.sin(2 * np.pi * freq * t)
        sd.play(sound_wave, samplerate=sample_rate, device=self.selected_audio_device)
        sd.wait()

    def trigger_stimulation(self, duration):
        try:
            task = Task()
            task.CreateAOVoltageChan(f"{self.daq_device}/ao0", "", -10.0, 10.0, 10348, None)
            sampling_rate = 5000
            total_samples = int(sampling_rate * duration)
            if total_samples <= 0:
                total_samples = int(sampling_rate * 0.1)
            output_data = np.full(total_samples, 5.0, dtype=np.float64)
            task.CfgSampClkTiming("", sampling_rate, 10280, 10178, total_samples)
            written = ctypes.c_int32()
            task.WriteAnalogF64(total_samples, False, 10.0, 0, output_data, ctypes.byref(written), None)
            task.StartTask()
            task.WaitUntilTaskDone(max(10.0, duration))
            task.StopTask()
            task.ClearTask()

            reset_task = Task()
            reset_task.CreateAOVoltageChan(f"{self.daq_device}/ao0", "", -10.0, 10.0, 10348, None)
            reset_task.StartTask()
            reset_task.WriteAnalogScalarF64(True, 0.001, 0.0, None)
            reset_task.StopTask()
            reset_task.ClearTask()
            
            print("Shock delivered")
        except Exception as e:
            print(f"Stimulation error: {e}")

    def on_closing(self):
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            for camera in self.cameras:
                if camera.previewing is True:
                    camera.stop_preview()
                if camera.recording is True:
                    camera.stop_record()
                camera.close()

            if self.arduino and self.arduino.is_open:
                self.arduino.close()
                print("Arduino port closed.")
                
            if self.speed_encoder_gui and self.speed_encoder_gui.running:
                self.speed_encoder_gui.stop_tracking()
                self.speed_encoder_gui.stop_update_loop()

            self.root.destroy()
            os._exit(0)
        
    def exit(self):
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            for camera in self.cameras:
                if camera.previewing is True:
                    camera.stop_preview()
                if camera.recording is True:
                    camera.stop_record()
                camera.close()
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
                print("Arduino port closed.")
                
            if self.speed_encoder_gui and self.speed_encoder_gui.running:
                self.speed_encoder_gui.stop_tracking()
                self.speed_encoder_gui.stop_update_loop()
                
            self.root.destroy()
            os._exit(0)
    
    def trigger_laser(self, frequency=20.0, pulse_width=10.0, duration=1000.0):
        try:
            if self.arduino and self.arduino.is_open:
                pulse_width_us = int(pulse_width * 1000)
                command = f"O,{int(frequency)},{pulse_width_us},{int(duration)}\n"
                self.arduino.write(command.encode())
                print(f"Laser stimulation triggered: {frequency}Hz, {pulse_width}ms, {duration}ms")
        except Exception as e:
            print(f"Laser stimulation error: {e}")

PYDAQMX_AVAILABLE = False
try:
    import PyDAQmx
    from PyDAQmx import Task
    PYDAQMX_AVAILABLE = True
except ImportError:
    PYDAQMX_AVAILABLE = False
    print("Warning: PyDAQmx library not found, DAQ functionality will be unavailable.")

class SetupDialog:
    def __init__(self, parent):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Device Setting")
        self.dialog.geometry("350x350")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
        self.ui_queue = queue.Queue()
        
        self.arduino_port = None
        self.daq_device = None
        self.skip_check = tk.BooleanVar(value=False)
        
        self.arduino_default = "COM7"
        self.daq_default = "Dev1"
        
        self.use_arduino = tk.BooleanVar(value=True)
        self.use_daq = tk.BooleanVar(value=True)
        
        self.arduino_refreshing = False
        self.daq_refreshing = False
        
        self.setup_ui()
        
        self.process_ui_queue()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        arduino_frame = ttk.LabelFrame(main_frame, text="Arduino Setting")
        arduino_frame.pack(fill=tk.X, padx=5, pady=5)
        
        arduino_check_frame = ttk.Frame(arduino_frame)
        arduino_check_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(arduino_check_frame, text="Use Arduino", variable=self.use_arduino, 
                       command=self.toggle_arduino_widgets).pack(side=tk.LEFT)
        
        port_frame = ttk.Frame(arduino_frame)
        port_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(port_frame, text="COM Port:").pack(side=tk.LEFT, padx=5)
        self.arduino_combo = ttk.Combobox(port_frame, state="readonly", width=15)
        self.arduino_combo.pack(side=tk.LEFT, padx=5)
        
        self.arduino_refresh_icon = ttk.Label(port_frame, text="↻", cursor="hand2")
        self.arduino_refresh_icon.pack(side=tk.LEFT, padx=5)
        self.arduino_refresh_icon.bind("<Button-1>", lambda e: self.refresh_ports())
        
        self.arduino_progress = ttk.Progressbar(port_frame, mode='indeterminate', maximum=20, length=20)
        self.arduino_progress.pack(side=tk.LEFT, padx=5)
        self.arduino_progress.pack_forget()
        
        self.arduino_status = ttk.Label(arduino_frame, text="")
        self.arduino_status.pack(fill=tk.X, padx=5, pady=2)
        
        daq_frame = ttk.LabelFrame(main_frame, text="DAQ Setting")
        daq_frame.pack(fill=tk.X, padx=5, pady=5)
        
        daq_check_frame = ttk.Frame(daq_frame)
        daq_check_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(daq_check_frame, text="Use DAQ", variable=self.use_daq, 
                       command=self.toggle_daq_widgets).pack(side=tk.LEFT)
        
        device_frame = ttk.Frame(daq_frame)
        device_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(device_frame, text="Device Name:").pack(side=tk.LEFT, padx=5)
        self.daq_combo = ttk.Combobox(device_frame, state="readonly", width=15)
        self.daq_combo.pack(side=tk.LEFT, padx=5)
        
        self.daq_refresh_icon = ttk.Label(device_frame, text="↻", cursor="hand2")
        self.daq_refresh_icon.pack(side=tk.LEFT, padx=5)
        self.daq_refresh_icon.bind("<Button-1>", lambda e: self.refresh_daq())
        
        self.daq_progress = ttk.Progressbar(device_frame, mode='indeterminate', maximum=20, length=20)
        self.daq_progress.pack(side=tk.LEFT, padx=5)
        self.daq_progress.pack_forget()
        
        self.daq_status = ttk.Label(daq_frame, text="")
        self.daq_status.pack(fill=tk.X, padx=5, pady=2)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel, width=10).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(btn_frame, text="Skip", command=self.skip_and_use_defaults, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(btn_frame, text="Confirm", command=self.on_ok, width=10).grid(row=0, column=2, padx=5, pady=5)
        
        self.status_label = ttk.Label(main_frame, text="Please select a device and click OK.")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        self.refresh_ports()
        self.refresh_daq()
    
    def toggle_arduino_widgets(self):
        state = "normal" if self.use_arduino.get() else "disabled"
        self.arduino_combo.config(state=state)
        self.arduino_refresh_icon.config(state=state)
    
    def toggle_daq_widgets(self):
        state = "normal" if self.use_daq.get() else "disabled"
        self.daq_combo.config(state=state)
        self.daq_refresh_icon.config(state=state)
    
    def process_ui_queue(self):
        try:
            while True:
                callback, args = self.ui_queue.get_nowait()
                callback(*args)
        except queue.Empty:
            pass
        
        self.dialog.after(100, self.process_ui_queue)
    
    def get_daq_devices(self):
        if not PYDAQMX_AVAILABLE:
            return ["Dev1", "Dev2", "Dev3"]
        
        try:
            device_names = PyDAQmx.create_string_buffer(1024)
            PyDAQmx.DAQmxGetSysDevNames(device_names, 1024)
            
            devices = device_names.value.decode('utf-8').split(',')
            
            return [dev for dev in devices if dev]
        except Exception as e:
            print(f"Error occurred while obtaining DAQ device: {e}")
            return []
    
    def refresh_ports(self):
        if self.arduino_refreshing:
            return
            
        self.arduino_refresh_icon.pack_forget()
        self.arduino_progress.pack(side=tk.LEFT, padx=5)
        self.arduino_progress.start(10)
        self.arduino_status.config(text="Scanning ports...")
        self.arduino_refreshing = True
        
        threading.Thread(target=self._refresh_ports_thread, daemon=True).start()
    
    def _refresh_ports_thread(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        
        self.ui_queue.put((self._update_ports_ui, [ports]))
    
    def _update_ports_ui(self, ports):
        self.arduino_combo['values'] = ports
        if ports:
            self.arduino_combo.current(0)
            self.arduino_status.config(text=f"Found {len(ports)} port(s)")
        else:
            self.arduino_status.config(text="No COM ports found")
        
        self.arduino_progress.stop()
        self.arduino_progress.pack_forget()
        self.arduino_refresh_icon.pack(side=tk.LEFT, padx=5)
        self.arduino_refreshing = False
    
    def refresh_daq(self):
        if self.daq_refreshing:
            return
            
        self.daq_refresh_icon.pack_forget()
        self.daq_progress.pack(side=tk.LEFT, padx=5)
        self.daq_progress.start(10)
        self.daq_status.config(text="Scanning devices...")
        self.daq_refreshing = True
        
        threading.Thread(target=self._refresh_daq_thread, daemon=True).start()
    
    def _refresh_daq_thread(self):
        daq_devices = self.get_daq_devices()
        
        self.ui_queue.put((self._update_daq_ui, [daq_devices]))
    
    def _update_daq_ui(self, daq_devices):
        self.daq_combo['values'] = daq_devices
        if daq_devices:
            self.daq_combo.current(0)
            self.daq_status.config(text=f"Found {len(daq_devices)} device(s)")
        else:
            self.daq_combo.set("")
            self.daq_status.config(text="No DAQ devices found")
        
        self.daq_progress.stop()
        self.daq_progress.pack_forget()
        self.daq_refresh_icon.pack(side=tk.LEFT, padx=5)
        self.daq_refreshing = False
    
    def skip_and_use_defaults(self):
        self.arduino_port = self.arduino_combo.get() or self.arduino_default
        self.daq_device = self.daq_combo.get() or self.daq_default
        
        self.skip_check.set(True)
        
        self.dialog.destroy()
        
    def on_ok(self):
        if self.use_arduino.get():
            self.arduino_port = self.arduino_combo.get()
            if not self.arduino_port:
                self.status_label.config(text="Please select the Arduino COM port.")
                return
        else:
            self.arduino_port = self.arduino_default
            
        if self.use_daq.get():
            self.daq_device = self.daq_combo.get()
            if not self.daq_device or self.daq_device == "No DAQ devices found":
                self.status_label.config(text="Please select the DAQ device.")
                return
        else:
            self.daq_device = self.daq_default
        
        if not self.skip_check.get():
            if self.use_arduino.get():
                self.arduino_status.config(text="Connecting to Arduino...")
                self.dialog.update()
                
                try:
                    arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
                    time.sleep(2)
                    arduino.close()
                    self.arduino_status.config(text="Arduino connected successfully")
                except Exception as e:
                    self.arduino_status.config(text=f"Arduino connection failed: {e}")
                    self.status_label.config(text=f"Arduino connection failed: {e}")
                    return
            
            if self.use_daq.get():
                self.daq_status.config(text="Connecting to DAQ device...")
                self.dialog.update()
                
                try:
                    if PYDAQMX_AVAILABLE:
                        task = Task()
                        task.CreateAOVoltageChan(f"{self.daq_device}/ao1", "", -10.0, 10.0, 10348, None)
                        task.ClearTask()
                        self.daq_status.config(text="DAQ device connected successfully")
                    else:
                        self.daq_status.config(text="PyDAQmx not available, skipping DAQ check")
                except Exception as e:
                    self.daq_status.config(text=f"DAQ device connection failed: {e}")
                    self.status_label.config(text=f"DAQ device connection failed: {e}")
                    return
            else:
                self.daq_status.config(text="Using default DAQ device, skipping check")
        
        self.status_label.config(text="All devices connected successfully!")
        self.dialog.after(1000, self.dialog.destroy)
        
    def on_cancel(self):
        self.arduino_port = None
        self.daq_device = None
        self.dialog.destroy()

def get_camera_names():
    """
    Get the friendly names of all connected video capture devices.
    """
    wmi = win32com.client.GetObject("winmgmts:")
    cameras = wmi.InstancesOf("Win32_PnPEntity")
    camera_list = []
    for camera in cameras:
        if camera.Service == "usbvideo":  # Only list USB video devices
            camera_list.append(camera.Name)
    return camera_list

def is_camera_available(index):
    """
    Robustly check if a camera is available and return (True/False, device name).
    """
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return False, None
        ret, _ = cap.read()
        cap.release()
        if ret:
            names = get_camera_names()
            if index < len(names):
                return True, names[index]
            else:
                return True, f"Camera {index}"
        else:
            return False, None
    except Exception as e:
        print(f"[Camera {index}] check failed: {e}")
        return False, None

def list_available_cameras(max_devices=10):
    """
    List available and accessible camera device names up to a maximum number.
    """
    available = []
    for i in range(max_devices):
        success, name = is_camera_available(i)
        if success and name:
            available.append(name)
    return available

def list_audio_devices():
    devices = sd.query_devices()
    return [d['name'] for d in devices if d['max_output_channels'] > 0]

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    setup_dialog = SetupDialog(root)
    root.wait_window(setup_dialog.dialog)
    
    if setup_dialog.arduino_port is not None and setup_dialog.daq_device is not None:
        root.deiconify()
        app = ExperimentGUI(root, setup_dialog.arduino_port, setup_dialog.daq_device)
        root.mainloop()
    else:
        root.destroy()