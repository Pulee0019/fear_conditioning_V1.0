# -*- coding: utf-8 -*-
""" 
Created on Fri Sep 5 20:50:50 2025 

This custom code involved the analysis of freezing and freezing + fiber through the process of deeplabcut.

Edit on Sat Dec 6 10:11:23 2025

Adjust the event code process to include 3, 4, 5 for optogenetics

@author: Pulee 
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
import struct
import re
import os
import math
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
import fnmatch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import json
from scipy.interpolate import interp1d
from itertools import groupby
import traceback
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Starting a Matplotlib GUI outside of the main thread")

import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

invert = True

# Global channel memory
CHANNEL_MEMORY_FILE = "channel_memory.json"
channel_memory = {}

def load_channel_memory():
    """Load channel memory from file"""
    global channel_memory
    if os.path.exists(CHANNEL_MEMORY_FILE):
        try:
            with open(CHANNEL_MEMORY_FILE, 'r') as f:
                channel_memory = json.load(f)
        except:
            channel_memory = {}

def save_channel_memory():
    """Save channel memory to file"""
    try:
        with open(CHANNEL_MEMORY_FILE, 'w') as f:
            json.dump(channel_memory, f)
    except:
        pass

load_channel_memory()

class ModeSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Lab Fear Conditioning Analyzer")
        self.root.geometry("500x320")
        self.root.resizable(False, False)
        
        self.root.configure(bg="#f0f0f0")
        
        main_frame = ttk.Frame(root, padding=40)
        main_frame.pack(expand=True, fill="both")
        
        title_label = ttk.Label(main_frame, text="ML Lab Fear Conditioning Analyzer", 
                               font=("Arial", 18, "bold"))
        title_label.pack(pady=20)
        
        subtitle_label = ttk.Label(main_frame, text="Please Select Your Analysis Mode!", 
                                 font=("Arial", 12))
        subtitle_label.pack(pady=10)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        freezing_btn = ttk.Button(button_frame, text="Freezing Analysis", 
                                 command=lambda: self.select_mode("freezing"),
                                 width=20)
        freezing_btn.grid(row=0, column=0, padx=10, pady=10)
        
        pupil_btn = ttk.Button(button_frame, text="Pupil Analysis", 
                              command=lambda: self.select_mode("pupil"),
                              width=20)
        pupil_btn.grid(row=0, column=1, padx=10, pady=10)
        
        version_label = ttk.Label(main_frame, text="Version 4.0 | © 2025 ML Lab", 
                                 font=("Arial", 8))
        version_label.pack(side="bottom", pady=5)
        
    def select_mode(self, mode):
        response = messagebox.askyesno("Fiber Photometry", 
                                      "Do you want to include fiber photometry analysis?")
        
        self.root.destroy()
        
        root = tk.Tk()
        root.geometry("1200x1200")
        root.title("Fear Conditioning Analyzer")
        
        if mode == "freezing":
            app = FreezingAnalyzerApp(root, include_fiber=response)
        else:
            app = PupilAnalyzerApp(root, include_fiber=response)
        
        root.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(root))
        root.mainloop()
    
    def on_closing(self, root):
        root.quit()
        root.destroy()

class BaseAnalyzerApp:
    def __init__(self, root, include_fiber=False):
        self.root = root
        self.include_fiber = include_fiber
        self.fiber_data = None
        self.video_start_fiber = None
        self.video_end_fiber = None
        self.channels = {}
        self.primary_signal = None
        self.preprocessed_data = None
        self.dff_data = None
        self.zscore_data = None
        self.event_data = None
        self.fiber_cropped = None
        self.event_time_absolute = None
        self.channel_data = {}
        self.active_channels = []
        self.smooth_window = tk.IntVar(value=11)
        self.smooth_order = tk.IntVar(value=5)
        self.baseline_start = tk.DoubleVar(value=0)
        self.baseline_end = tk.DoubleVar(value=120)
        self.apply_smooth = tk.BooleanVar(value=False)
        self.apply_baseline = tk.BooleanVar(value=False)
        self.apply_motion = tk.BooleanVar(value=False)
        self.target_signal_var = tk.StringVar(value="470")
        self.reference_signal_var = tk.StringVar(value="410")
        self.baseline_period = [0, 120]
        self.baseline_model = tk.StringVar(value="Polynomial")
        self.multi_animal_data = []
        self.current_animal_index = 0
        self.selected_files = []
        
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)
        
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10, ipadx=5, ipady=5)
        
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Create notebook for group tabs
        self.plot_notebook = ttk.Notebook(self.plot_frame)
        self.plot_notebook.pack(fill="both", expand=True)
        
        # Create initial default tab
        self.default_tab = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.default_tab, text="Plots")
        
        self.default_fig = Figure(figsize=(8, 4), dpi=100)
        self.default_canvas = FigureCanvasTkAgg(self.default_fig, master=self.default_tab)
        self.default_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        default_toolbar_frame = ttk.Frame(self.default_tab)
        default_toolbar_frame.pack(fill="x")
        self.default_toolbar = NavigationToolbar2Tk(self.default_canvas, default_toolbar_frame)
        self.default_toolbar.update()
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
        
        self.create_ui()
        
    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def create_ui(self):
        for widget in self.control_frame.winfo_children():
            widget.destroy()
            
        if self.include_fiber:
            self.create_step_navigation()
        
        self.create_main_controls()
    
    def create_step_navigation(self):
        step_frame = ttk.LabelFrame(self.control_frame, text="Analysis Steps")
        step_frame.pack(fill="x", padx=5, pady=5)
        
        steps = ttk.Frame(step_frame)
        steps.pack(fill="x")
        steps.columnconfigure(0, weight=1)
        steps.columnconfigure(1, weight=1)
        
        ttk.Button(steps, text="1. Data Load", command=self.show_step1).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(steps, text="2. Preprocessing", command=self.show_step2).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(steps, text="3. ΔF/F & Z-score", command=self.show_step3).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(steps, text="4. Event Analysis", command=self.show_step4).grid(row=1, column=1, padx=2, pady=2, sticky="ew")    
    
    def create_main_controls(self):
        pass
    
    def single_import(self):
        """Single animal import - select one animal folder"""
        folder_path = filedialog.askdirectory(title="Select animal ear tag folder")
        if not folder_path:
            return

        exp_mode = "freezing+fiber" if self.include_fiber else "freezing"
        if "PupilAnalyzerApp" in str(type(self)):
            exp_mode = "pupil+fiber" if self.include_fiber else "pupil"
        
        try:
            folder_path = os.path.normpath(folder_path)
            path_parts = folder_path.split(os.sep)
            if len(path_parts) < 4:
                messagebox.showwarning("Invalid Directory", "Selected directory is not a valid animal data folder")
                return

            batch_name = path_parts[-3]
            group = path_parts[-2]
            ear_tag = path_parts[-1]
            animal_id = f"{batch_name}-{ear_tag}"

            if any(f"{d['group']}-{d['animal_id']}" == f"{group}-{animal_id}" for d in self.multi_animal_data):
                messagebox.showwarning("Animal Duplicate", f"Skip duplicate animal: {animal_id}")
                return
    
            patterns = {
                'dlc': ['*dlc*.csv', '*deeplabcut*.csv'],
                'events': ['*events*.csv', '*timeline*.csv'],
                'fiber': ['fluorescence.csv', '*fiber*.csv'],
                'timestamp': ['*timestamp*.csv', '*time*.csv'],
                'ast2': ['*.ast2']
            }

            required_files = {
                'freezing': ['dlc', 'events'],
                'freezing+fiber': ['dlc', 'events', 'fiber'],
                'pupil': ['dlc', 'timestamp', 'ast2'],
                'pupil+fiber': ['dlc', 'timestamp', 'ast2', 'fiber']
            }[exp_mode]

            files_found = {}
            for file_type, file_patterns in patterns.items():
                found_file = None
                for root_path, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_lower = file.lower()
                        for pattern in file_patterns:
                            if fnmatch.fnmatch(file_lower, pattern.lower()):
                                found_file = os.path.join(root_path, file)
                                files_found[file_type] = found_file
                                break
                        if found_file:
                            break
                    if found_file:
                        break

            missing_files = [ft for ft in required_files if ft not in files_found]
            if missing_files:
                messagebox.showwarning("Missing Files", 
                                    f"Required files missing: {', '.join(missing_files)}\nfor mode: {exp_mode}")
                return

            animal_data = {
                'animal_id': animal_id,
                'group': group,
                'files': files_found,
                'processed': False,
                'event_time_absolute': False,
                'active_channels': []
            }

            if 'events' in files_found:
                try:
                    event_data = pd.read_csv(files_found['events'])
                    start_offset = event_data['start_time'].iloc[0]
                    event_data['start_time'] = event_data['start_time'] - start_offset
                    event_data['end_time'] = event_data['end_time'] - start_offset
                    animal_data['event_data'] = event_data
                except Exception as e:
                    print(f"Failed to load events for {animal_id}: {str(e)}")

            if 'dlc' in files_found:
                try:
                    dlc_data = pd.read_csv(files_found['dlc'], header=[0,1,2])
                    animal_data['dlc_data'] = dlc_data
                    filename = os.path.basename(files_found['dlc'])
                    match = re.search(r'cam(\d+)', filename)
                    if match:
                        animal_data['cam_id'] = int(match.group(1))
                except Exception as e:
                    print(f"Failed to load DLC for {animal_id}: {str(e)}")

            if 'timestamp' in files_found:
                try:
                    timestamps = pd.read_csv(files_found['timestamp'])
                    exp_start = timestamps[(timestamps['Device'] == 'Experiment') &
                                        (timestamps['Action'] == 'Start')]['Timestamp'].values
                    if len(exp_start) > 0:
                        animal_data['experiment_start'] = exp_start[0]
                except Exception as e:
                    print(f"Failed to load timestamp for {animal_id}: {str(e)}")

            if 'ast2' in files_found:
                try:
                    ast2_data = self.h_AST2_readData(files_found['ast2'])
                    animal_data['ast2_data'] = ast2_data
                except Exception as e:
                    print(f"Failed to load AST2 for {animal_id}: {str(e)}")

            if 'fiber' in files_found:
                try:
                    fiber_result = self.load_fiber_data(files_found['fiber'])
                    if fiber_result:
                        animal_data.update(fiber_result)
                except Exception as e:
                    print(f"Failed to load fiber for {animal_id}: {str(e)}")

            animal_data['processed'] = True
            self.selected_files.append(animal_data)
            self.multi_animal_data.append(animal_data)
            self.file_listbox.insert(tk.END, f"{animal_id} ({group})")
            
            if self.include_fiber and 'fiber_data' in animal_data:
                self.show_channel_selection_dialog()
            
            self.set_status(f"Added animal: {animal_id} ({group})")
            messagebox.showinfo("Success", f"Successfully added animal:\n{animal_id} ({group})")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add single animal: {str(e)}")
            self.set_status("Add animal failed")

    def multi_import(self):
        """Multi-animal import - scan entire directory structure"""
        base_dir = filedialog.askdirectory(title="Select Free Moving/Behavioural directory")
        if not base_dir:
            return

        exp_mode = "freezing+fiber" if self.include_fiber else "freezing"
        if "PupilAnalyzerApp" in str(type(self)):
            exp_mode = "pupil+fiber" if self.include_fiber else "pupil"

        try:
            self.set_status("Scanning for multi-animal data...")
            self.selected_files = []
            self.multi_animal_data = []
            self.file_listbox.delete(0, tk.END)

            batch_dirs = glob.glob(os.path.join(base_dir, "20*"))
            for batch_dir in batch_dirs:
                if not os.path.isdir(batch_dir):
                    continue

                batch_name = os.path.basename(batch_dir)

                for group in ["FC", "EXT_1", "EXT_2", "RET", "Hab"]:
                    group_dir = os.path.join(batch_dir, group)
                    if not os.path.exists(group_dir):
                        continue

                    ear_tag_dirs = glob.glob(os.path.join(group_dir, "*"))
                    for ear_tag_dir in ear_tag_dirs:
                        if not os.path.isdir(ear_tag_dir):
                            continue

                        ear_tag = os.path.basename(ear_tag_dir)
                        animal_id = f"{batch_name}-{ear_tag}"

                        files_found = {}
                        patterns = {
                            'dlc': ['*dlc*.csv', '*deeplabcut*.csv'],
                            'events': ['*events*.csv', '*timeline*.csv'],
                            'fiber': ['fluorescence.csv', '*fiber*.csv'],
                            'timestamp': ['*timestamp*.csv', '*time*.csv'],
                            'ast2': ['*.ast2']
                        }

                        for file_type, file_patterns in patterns.items():
                            found_file = None
                            for root_path, dirs, files in os.walk(ear_tag_dir):
                                for file in files:
                                    file_lower = file.lower()
                                    for pattern in file_patterns:
                                        if fnmatch.fnmatch(file_lower, pattern.lower()):
                                            found_file = os.path.join(root_path, file)
                                            files_found[file_type] = found_file
                                            break
                                    if found_file:
                                        break
                                if found_file:
                                    break

                        animal_data = {
                            'animal_id': animal_id,
                            'group': group,
                            'files': files_found,
                            'processed': False,
                            'event_time_absolute': False,
                            'active_channels': []
                        }

                        required_files = ['dlc', 'events'] if 'freezing' in exp_mode else ['dlc', 'timestamp']
                        if all(ft in files_found for ft in required_files):
                            if any(f"{d['group']}-{d['animal_id']}" == f"{group}-{animal_id}" for d in self.multi_animal_data):
                                messagebox.showwarning("Animal Duplicate", f"Skip duplicate animal: {animal_id}")
                                continue

                            if 'events' in files_found:
                                try:
                                    event_data = pd.read_csv(files_found['events'])
                                    start_offset = event_data['start_time'].iloc[0]
                                    event_data['start_time'] = event_data['start_time'] - start_offset
                                    event_data['end_time'] = event_data['end_time'] - start_offset
                                    animal_data['event_data'] = event_data
                                except Exception as e:
                                    print(f"Failed to load events for {animal_id}: {str(e)}")

                            if 'dlc' in files_found:
                                try:
                                    dlc_data = pd.read_csv(files_found['dlc'], header=[0,1,2])
                                    animal_data['dlc_data'] = dlc_data
                                    filename = os.path.basename(files_found['dlc'])
                                    match = re.search(r'cam(\d+)', filename)
                                    if match:
                                        animal_data['cam_id'] = int(match.group(1))
                                except Exception as e:
                                    print(f"Failed to load DLC for {animal_id}: {str(e)}")

                            if 'timestamp' in files_found:
                                try:
                                    timestamps = pd.read_csv(files_found['timestamp'])
                                    exp_start = timestamps[(timestamps['Device'] == 'Experiment') &
                                                        (timestamps['Action'] == 'Start')]['Timestamp'].values
                                    if len(exp_start) > 0:
                                        animal_data['experiment_start'] = exp_start[0]
                                except Exception as e:
                                    print(f"Failed to load timestamp for {animal_id}: {str(e)}")

                            if 'ast2' in files_found:
                                try:
                                    ast2_data = self.h_AST2_readData(files_found['ast2'])
                                    animal_data['ast2_data'] = ast2_data
                                except Exception as e:
                                    print(f"Failed to load AST2 for {animal_id}: {str(e)}")

                            if 'fiber' in files_found:
                                try:
                                    fiber_result = self.load_fiber_data(files_found['fiber'])
                                    if fiber_result:
                                        animal_data.update(fiber_result)
                                except Exception as e:
                                    print(f"Failed to load fiber for {animal_id}: {str(e)}")
                            
                            animal_data['processed'] = True
                            self.multi_animal_data.append(animal_data)
                            self.selected_files.append(animal_data)
                            self.file_listbox.insert(tk.END, f"{animal_id} ({group})")

            if not self.selected_files:
                messagebox.showwarning("No Data", "No valid animal data found in the selected directory")
                self.set_status("No multi-animal data found")
            else:
                self.set_status(f"Found {len(self.selected_files)} animals")
                if self.include_fiber:
                    self.show_channel_selection_dialog()
                else:
                    messagebox.showinfo("Success", f"Found and processed {len(self.selected_files)} animals")
                    self.set_status(f"Imported {len(self.selected_files)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")
            self.set_status("Import failed")

    def clear_selected(self):
        """Clear selected animals from the list"""
        selected_indices = self.file_listbox.curselection()
        for index in sorted(selected_indices, reverse=True):
            animal_id = self.file_listbox.get(index)
            self.file_listbox.delete(index)
            
            if index < len(self.selected_files):
                animal_data = self.selected_files.pop(index)
                for i, data in enumerate(self.multi_animal_data):
                    if data['animal_id'] == animal_data['animal_id']:
                        self.multi_animal_data.pop(i)
                        break
        self.set_status(f"Cleared {len(selected_indices)} animals")
    
    def select_all(self):
        """Select all animals in the list"""
        self.file_listbox.selection_set(0, tk.END)
        self.set_status(f"Selected all {self.file_listbox.size()} animals")

    def load_fiber_data(self, file_path):
        """Load fiber photometry data"""
        try:
            fiber_data = pd.read_csv(file_path, skiprows=1, delimiter=',')
            fiber_data = fiber_data.loc[:, ~fiber_data.columns.str.contains('^Unnamed')]
            fiber_data.columns = fiber_data.columns.str.strip()

            time_col = None
            possible_time_columns = ['timestamp', 'timems', 'time', 'time(ms)']
            for col in fiber_data.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in possible_time_columns):
                    time_col = col
                    break
            
            if not time_col:
                numeric_cols = fiber_data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    time_col = numeric_cols[0]
            
            fiber_data[time_col] = fiber_data[time_col] / 1000

            global fps_fiber
            fps_fiber = int(1 / np.mean(np.diff(fiber_data[time_col])))
            
            events_col = None
            for col in fiber_data.columns:
                if 'event' in col.lower():
                    events_col = col
                    break
            
            channels = {'time': time_col, 'events': events_col}
            
            channel_data = {}
            available_wavelengths = set()
            channel_pattern = re.compile(r'CH(\d+)-(\d+)', re.IGNORECASE)
            
            for col in fiber_data.columns:
                match = channel_pattern.match(col)
                if match:
                    channel_num = int(match.group(1))
                    wavelength = int(match.group(2))
                    
                    if channel_num not in channel_data:
                        channel_data[channel_num] = {}
                    
                    channel_data[channel_num][str(wavelength)] = col
                    if wavelength != 410:
                        available_wavelengths.add(str(wavelength))
                        
            wavelength_combinations = self.generate_wavelength_combinations(available_wavelengths)

            self.set_status(f"Fiber data loaded, {len(channel_data)} channels detected")
            
            return {
                'fiber_data': fiber_data,
                'channels': channels,
                'channel_data': channel_data,
                'available_wavelengths': sorted(available_wavelengths),
                'wavelength_combinations': wavelength_combinations
            }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load fiber data: {str(e)}")
            self.set_status("Fiber data load failed")
            return None
    
    def generate_wavelength_combinations(self, wavelengths):
        from itertools import combinations
        wavelength_list = sorted(wavelengths)
        combinations_list = []
        
        for wl in wavelength_list:
            combinations_list.append(wl)
        
        for r in range(2, min(4, len(wavelength_list) + 1)):
            for combo in combinations(wavelength_list, r):
                combinations_list.append('+'.join(combo))
        
        return combinations_list

    def show_channel_selection_dialog(self):
        """Show dialog for channel selection with memory"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Channels")
        dialog.geometry("290x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=5)
        main_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(main_frame)
        canvas.config(height=50, width=100)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.channel_vars = {}
        
        for animal_data in self.multi_animal_data:
            if 'channel_data' not in animal_data:
                continue
                
            animal_id = animal_data['animal_id']
            group = animal_data.get('group', '')
            label = f"{animal_id} ({group})" if group else animal_id
            
            animal_frame = ttk.LabelFrame(scrollable_frame, text=label)
            animal_frame.pack(fill="x", padx=5, pady=5, ipadx=5, ipady=5)
            
            self.channel_vars[animal_id] = {}
            
            # Load channel memory for this animal
            remembered_channels = channel_memory.get(animal_id, [1])
            
            for channel_num in sorted(animal_data['channel_data'].keys()):
                var = tk.BooleanVar(value=(channel_num in remembered_channels))
                self.channel_vars[animal_id][channel_num] = var
                
                chk = ttk.Checkbutton(
                    animal_frame, 
                    text=f"Channel {channel_num}",
                    variable=var
                )
                chk.pack(anchor="w", padx=2, pady=2)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="Select All", command=lambda: self.toggle_all_channels(True)).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Deselect All", command=lambda: self.toggle_all_channels(False)).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Confirm", command=lambda: self.finalize_channel_selection(dialog)).grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        
    def toggle_all_channels(self, select):
        """Toggle all channel selections"""
        for animal_vars in self.channel_vars.values():
            for var in animal_vars.values():
                var.set(select)

    def finalize_channel_selection(self, dialog):
        """Finalize channel selection and save to memory"""
        global channel_memory
        
        for animal_data in self.multi_animal_data:
            animal_id = animal_data['animal_id']
            
            if animal_id in self.channel_vars:
                selected_channels = []
                for channel_num, var in self.channel_vars[animal_id].items():
                    if var.get():
                        selected_channels.append(channel_num)
                
                if not selected_channels:
                    messagebox.showwarning(
                        "No Channels Selected", 
                        f"Please select at least one channel for {animal_id}"
                    )
                    return
                    
                animal_data['active_channels'] = selected_channels
                # Save to memory
                channel_memory[animal_id] = selected_channels
                self.set_video_markers(animal_data)

        dialog.destroy()
        save_channel_memory()
        messagebox.showinfo("Success", f"Processed {len(self.multi_animal_data)} animals")
        self.set_status(f"Imported {len(self.multi_animal_data)} animals")
    
    def set_video_markers(self, animal_data):
        """Set video markers for specific animal"""
        try:
            fiber_data = animal_data.get('fiber_data')
            event_data = animal_data.get('event_data')
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            
            if fiber_data is None or not active_channels:
                return
            
            events_col = channels.get('events')
            if events_col is None or events_col not in fiber_data.columns:
                messagebox.showerror("Error", "Events column not found in fiber data")
                return
            
            input1_events = fiber_data[fiber_data[events_col].str.contains('Input1', na=False)]
            if len(input1_events) < 2:
                messagebox.showerror("Error", "Could not find enough Input1 events (need at least 2)")
                return
            
            time_col = channels['time']
            video_start_fiber = input1_events[time_col].iloc[0]
            video_end_fiber = input1_events[time_col].iloc[-2]
            global video_duration
            video_duration = video_end_fiber - video_start_fiber
            
            fiber_cropped = fiber_data[
                (fiber_data[time_col] >= video_start_fiber) & 
                (fiber_data[time_col] <= video_end_fiber)].copy()
            
            animal_data.update({
                'fiber_cropped': fiber_cropped,
                'video_start_fiber': video_start_fiber,
                'video_end_fiber': video_end_fiber
            })
            
            if event_data is not None and not animal_data.get('event_time_absolute'):
                event_data['start_absolute'] = video_start_fiber + event_data['start_time']
                event_data['end_absolute'] = video_start_fiber + event_data['end_time']
                animal_data['event_time_absolute'] = True
            
            self.set_status(f"Video markers set for {animal_data['animal_id']}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set video markers: {str(e)}")
            self.set_status("Video markers failed")

    def plot_raw_data(self):
        """Plot raw fiber data for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        selected_animals = [self.multi_animal_data[i] for i in selected_indices]
        
        try:
            raw_data_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "raw data":
                    raw_data_tab_index = i
                    break
            
            if raw_data_tab_index is not None:
                self.plot_notebook.forget(raw_data_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
            
            raw_data_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(raw_data_tab, text="raw data")
            
            # Group animals by group
            groups = {}
            for animal_data in selected_animals:
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            group_notebook = ttk.Notebook(raw_data_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Create tabs for each group within the raw data tab
            for group_name, group_animals in groups.items():
                # Create group tab
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create notebook for animals within this group
                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Create subplots for each animal in the group
                for animal_data in group_animals:
                    if 'fiber_cropped' not in animal_data:
                        continue
                    
                    animal_id = animal_data['animal_id']
                    
                    # Create tab for this animal
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)
                    
                    # Create figure and canvas for this animal
                    fig = Figure(figsize=(8, 4), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(animal_tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax = fig.add_subplot(111)
                    
                    # Get time data
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['fiber_cropped'][time_col] - animal_data['video_start_fiber']
                    
                    # Plot all channels for this animal
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            for wavelength, col_name in animal_data['channel_data'][channel_num].items():
                                if col_name and col_name in animal_data['fiber_cropped'].columns:
                                    # Use different colors for different wavelengths
                                    if wavelength == "410":
                                        color = "#0000FF"
                                        alpha = 0.7
                                    elif wavelength == "470":
                                        color = "#00FF00" 
                                        alpha = 0.7
                                    elif wavelength == "560":
                                        color = "#FF0000"
                                        alpha = 0.7
                                    else:
                                        color = "#888888"
                                        alpha = 0.7
                                    
                                    # Highlight target signal
                                    if wavelength == self.target_signal_var.get():
                                        linewidth = 1.5
                                    else:
                                        linewidth = 1.0
                                    
                                    ax.plot(time_data, animal_data['fiber_cropped'][col_name], 
                                        color=color, linewidth=linewidth, alpha=alpha, 
                                        label=f'CH{channel_num} {wavelength}')
                    
                    ax.set_title(f"Raw Fiber Photometry Data - {animal_id}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Time (s)", fontsize=12)
                    ax.set_xlim([time_data.min(), time_data.max()])
                    ax.set_ylabel("Signal Intensity", fontsize=12)
                    ax.tick_params(axis='both', labelsize=10)
                    ax.grid(False)
                    ax.legend()
                    
                    # Add event markers if available
                    if 'event_data' in animal_data:
                        event_data = animal_data['event_data']
                        event_time_absolute = animal_data.get('event_time_absolute', False)
                        if event_time_absolute:
                            start_col = 'start_absolute'
                            end_col = 'end_absolute'
                        else:
                            start_col = 'start_time'
                            end_col = 'end_time'
                        
                        video_start = animal_data['video_start_fiber']
                        
                        for _, row in event_data.iterrows():
                            if row['Event Type'] == 0:
                                color = "#ffffff"
                            elif row['Event Type'] == 3:
                                color = "#ffffff"
                            else:
                                color = "#fc0000"

                            start_time = row[start_col] - video_start
                            end_time = row[end_col] - video_start
                            ax.axvspan(start_time, end_time, color=color, alpha=0.5)
                    
                    canvas.draw()
                
            self.set_status("Raw fiber data plotted with group-animal visualization")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raw data: {str(e)}")
            self.set_status("Raw plot failed")
    
    def get_target_columns(self, animal_data, target_signal):
        """Get target columns for each wavelength separately
        Returns: list of (channel_num, wavelength, col_name) tuples
        """
        target_columns = []
        
        if '+' in target_signal:
            # For combined wavelengths, return each wavelength separately
            wavelengths = target_signal.split('+')
            for channel_num in animal_data['active_channels']:
                if channel_num in animal_data['channel_data']:
                    for wl in wavelengths:
                        col = animal_data['channel_data'][channel_num].get(wl)
                        if col and col in animal_data.get('preprocessed_data', animal_data.get('fiber_cropped', pd.DataFrame())).columns:
                            target_columns.append((channel_num, col, wl))
        else:
            # For single wavelength
            for channel_num in animal_data['active_channels']:
                if channel_num in animal_data['channel_data']:
                    col = animal_data['channel_data'][channel_num].get(target_signal)
                    if col and col in animal_data.get('preprocessed_data', animal_data.get('fiber_cropped', pd.DataFrame())).columns:
                        target_columns.append((channel_num, col, target_signal))
        
        return target_columns

    def smooth_data(self):
        """Apply smoothing to all signals (target + 410) for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            window_size = self.smooth_window.get()
            poly_order = self.smooth_order.get()
            target_signal = self.target_signal_var.get()
            
            if window_size % 2 == 0:
                window_size += 1
            
            smooth_data_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "smooth data":
                    smooth_data_tab_index = i
                    break
            
            if smooth_data_tab_index is not None:
                self.plot_notebook.forget(smooth_data_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)

            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            smooth_data_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(smooth_data_tab, text="smooth data")
            
            group_notebook = ttk.Notebook(smooth_data_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Apply smoothing and plot by group and animal
            for group_name, group_animals in groups.items():
                # Create group tab
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create notebook for animals within this group
                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Create subplots for each animal in the group
                for animal_data in group_animals:
                    animal_id = animal_data['animal_id']
                    
                    # Create tab for this animal
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)
                    
                    # Create figure and canvas for this animal
                    fig = Figure(figsize=(8, 4), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(animal_tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax1 = fig.add_subplot(2, 1, 1)
                    ax2 = fig.add_subplot(2, 1, 2)
                    
                    animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()

                    # Get all wavelengths to process (target + 410)
                    wavelengths_to_process = set()
                    target_columns = self.get_target_columns(animal_data, target_signal)
                    ref_columns = self.get_target_columns(animal_data, "410")
                    
                    for _, _, wavelength in target_columns:
                        wavelengths_to_process.add(wavelength)
                    for _, _, wavelength in ref_columns:
                        wavelengths_to_process.add(wavelength)

                    wavelength_colors = {
                        '410': '#0000FF',  # Blue
                        '470': '#00FF00',  # Green
                        '560': '#FF0000',  # Red
                    }

                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']

                    # Process each wavelength
                    for channel_num in animal_data['active_channels']:
                        if channel_num not in animal_data['channel_data']:
                            continue
                        
                        for wavelength in wavelengths_to_process:
                            cols = animal_data['channel_data'][channel_num].get(wavelength)
                            if not cols or cols not in animal_data['fiber_cropped'].columns:
                                continue
                            
                            signal_data = animal_data['preprocessed_data'][cols]
                            smoothed_col = f"CH{channel_num}_{wavelength}_smoothed"
                            animal_data['preprocessed_data'][smoothed_col] = savgol_filter(
                                signal_data, window_size, poly_order)
                            
                            color = wavelength_colors.get(wavelength, "#808080")
                            
                            # Plot raw
                            ax1.plot(time_data, signal_data, 
                                color=color, linewidth=2, alpha=0.5, linestyle='--',
                                label=f'CH{channel_num} {wavelength}nm raw')
                            
                            ax2.plot(time_data, signal_data, 
                                color=color, linewidth=2, alpha=0.5, linestyle='--',
                                label=f'CH{channel_num} {wavelength}nm raw')
                            
                            # Plot smoothed
                            ax1.plot(time_data, animal_data['preprocessed_data'][smoothed_col], 
                                color=color, linewidth=2, alpha=0.8, linestyle='-',
                                label=f'CH{channel_num} {wavelength}nm smoothed')
                            
                            ax2.plot(time_data, animal_data['preprocessed_data'][smoothed_col], 
                                color=color, linewidth=2, alpha=0.8, linestyle='-',
                                label=f'CH{channel_num} {wavelength}nm smoothed')
                    
                    ax1.set_title(f"Smoothed Data - {animal_id} - window size={window_size} - poly order={poly_order}",
                                  fontsize=14, fontweight='bold')
                    ax1.set_xlim([time_data.min(), time_data.max()])
                    ax1.set_ylabel("Signal Intensity", fontsize=12)
                    ax1.tick_params(axis='both', labelsize=10)
                    ax1.grid(False)
                    ax1.legend()
                    
                    ax2.set_xlabel("Time (s)", fontsize=12)
                    ax2.set_xlim([45, 80])
                    ax2.set_ylim([45, 50])
                    ax2.set_ylabel("Signal Intensity", fontsize=12)
                    ax2.tick_params(axis='both', labelsize=10)
                    ax2.grid(False)
                    ax2.legend()
                    
                    # Add event markers if available
                    if 'event_data' in animal_data:
                        event_data = animal_data['event_data']
                        event_time_absolute = animal_data.get('event_time_absolute', False)
                        if event_time_absolute:
                            start_col = 'start_absolute'
                            end_col = 'end_absolute'
                        else:
                            start_col = 'start_time'
                            end_col = 'end_time'
                        
                        video_start = animal_data['video_start_fiber']
                        
                        for _, row in event_data.iterrows():
                            if row['Event Type'] == 0:
                                color = "#ffffff"
                            elif row['Event Type'] == 3:
                                color = "#ffffff"
                            else:
                                color = "#fc0000"
                            start_time = row[start_col] - video_start
                            end_time = row[end_col] - video_start
                            ax1.axvspan(start_time, end_time, color=color, alpha=0.5)

                    canvas.draw()
                
            self.set_status(f"Data smoothed with window={window_size}, order={poly_order}, all signals")
            messagebox.showinfo("Success", f"Smoothing applied to {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Smoothing failed: {str(e)}")
            self.set_status("Smoothing failed")

    def baseline_correction(self):
        """Apply baseline correction to all signals (target + 410) for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            model_type = self.baseline_model.get()
            target_signal = self.target_signal_var.get()
            
            baseline_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "baseline correction":
                    baseline_tab_index = i
                    break
            
            if baseline_tab_index is not None:
                self.plot_notebook.forget(baseline_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
                
            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            baseline_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(baseline_tab, text="baseline correction")
            
            group_notebook = ttk.Notebook(baseline_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Apply baseline correction and plot by group and animal
            for group_name, group_animals in groups.items():
                # Create group tab
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create notebook for animals within this group
                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Create subplots for each animal in the group
                for animal_data in group_animals:
                    animal_id = animal_data['animal_id']
                    
                    # Create tab for this animal
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)
                    
                    # Create figure and canvas for this animal
                    fig = Figure(figsize=(8, 4), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(animal_tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax1 = fig.add_subplot(2, 1, 1)
                    ax2 = fig.add_subplot(2, 1, 2)
                    
                    if animal_data.get('preprocessed_data') is None:
                        animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()
                        
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    full_time = time_data.values

                    # Get all wavelengths to process (target + 410)
                    wavelengths_to_process = set()
                    target_columns = self.get_target_columns(animal_data, target_signal)
                    ref_columns = self.get_target_columns(animal_data, "410")
                    
                    for _, _, wavelength in target_columns:
                        wavelengths_to_process.add(wavelength)
                    for _, _, wavelength in ref_columns:
                        wavelengths_to_process.add(wavelength)

                    wavelength_colors = {
                        '410': '#0000FF',
                        '470': '#00FF00',
                        '560': '#FF0000',
                    }
                    
                    # Process each wavelength
                    for channel_num in animal_data['active_channels']:
                        if channel_num not in animal_data['channel_data']:
                            continue
                        
                        for wavelength in wavelengths_to_process:
                            cols = animal_data['channel_data'][channel_num].get(wavelength)
                            if not cols or cols not in animal_data['preprocessed_data'].columns:
                                continue
                            
                            # Get signal data
                            if f"CH{channel_num}_{wavelength}_smoothed" in animal_data['preprocessed_data'].columns:
                                signal_col = f"CH{channel_num}_{wavelength}_smoothed"
                                signal_data = animal_data['preprocessed_data'][signal_col].values
                            else:
                                signal_data = animal_data['preprocessed_data'][cols].values
                            
                            # Fit baseline
                            if model_type.lower() == "exponential":
                                def exp_model(t, a, b, c):
                                    return a * np.exp(-b * t) + c
                                
                                p0 = [
                                    np.max(signal_data) - np.min(signal_data),
                                    0.01,
                                    np.min(signal_data)
                                ]
                                
                                try:
                                    params, _ = curve_fit(exp_model, full_time, signal_data, p0=p0, maxfev=5000)
                                    baseline_pred = exp_model(full_time, *params)
                                except Exception as e:
                                    X = full_time.reshape(-1, 1)
                                    model = LinearRegression()
                                    model.fit(X, signal_data)
                                    baseline_pred = model.predict(X)
                            else:
                                X = full_time.reshape(-1, 1)
                                model = LinearRegression()
                                model.fit(X, signal_data)
                                baseline_pred = model.predict(X)
                            
                            # Store baseline corrected data
                            baseline_corrected_col = f"CH{channel_num}_{wavelength}_baseline_corrected"
                            animal_data['preprocessed_data'][baseline_corrected_col] = signal_data - baseline_pred
                            animal_data['preprocessed_data'][f"CH{channel_num}_{wavelength}_baseline_pred"] = baseline_pred
                            
                            color = wavelength_colors.get(wavelength, "#808080")
                            
                            # Plot original signal
                            if f"CH{channel_num}_{wavelength}_smoothed" in animal_data['preprocessed_data'].columns:
                                ax1.plot(time_data, animal_data['preprocessed_data'][f"CH{channel_num}_{wavelength}_smoothed"], 
                                    color=color, linewidth=2,
                                    label=f'CH{channel_num} {wavelength}nm smoothed')
                            else:
                                ax1.plot(time_data, animal_data['preprocessed_data'][cols], 
                                    color=color, linewidth=2,
                                    label=f'CH{channel_num} {wavelength}nm raw')
                                
                            # Plot baseline corrected
                            ax2.plot(time_data, animal_data['preprocessed_data'][baseline_corrected_col], 
                                color=color, linewidth=2,
                                label=f'CH{channel_num} {wavelength}nm baseline corrected')
                            
                            # Plot baseline fit
                            ax1.plot(time_data, baseline_pred, 
                                color='k', linestyle='--', linewidth=1,
                                label=f'CH{channel_num} {wavelength}nm baseline fit')
                    
                    ax1.set_title(f"Baseline Corrected Signals - {animal_id} ({model_type} model)", fontsize=14, fontweight='bold')
                    ax1.set_ylabel("Signal Intensity", fontsize=12)
                    ax1.set_xlim([time_data.min(), time_data.max()])
                    ax1.tick_params(axis='both', labelsize=10)
                    ax1.grid(False)
                    ax1.legend()
                    ax2.set_xlabel("Time (s)", fontsize=12)
                    ax2.set_ylabel("Signal Intensity", fontsize=12)
                    ax2.set_xlim([time_data.min(), time_data.max()])
                    ax2.tick_params(axis='both', labelsize=10)
                    ax2.grid(False)
                    ax2.legend()

                    # Add event markers if available
                    if 'event_data' in animal_data:
                        event_data = animal_data['event_data']
                        event_time_absolute = animal_data.get('event_time_absolute', False)
                        if event_time_absolute:
                            start_col = 'start_absolute'
                            end_col = 'end_absolute'
                        else:
                            start_col = 'start_time'
                            end_col = 'end_time'
                        
                        video_start = animal_data['video_start_fiber']
                        
                        for _, row in event_data.iterrows():
                            if row['Event Type'] == 0:
                                color = "#ffffff"
                            elif row['Event Type'] == 3:
                                color = "#ffffff"
                            else:
                                color = "#fc0000"

                            start_time = row[start_col] - video_start
                            end_time = row[end_col] - video_start
                            ax1.axvspan(start_time, end_time, color=color, alpha=0.5)
                            ax2.axvspan(start_time, end_time, color=color, alpha=0.5)
                            
                    canvas.draw()
                
            self.set_status(f"Baseline correction applied ({model_type} model, all signals)")
            messagebox.showinfo("Success", f"Baseline correction applied to {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Baseline correction failed: {str(e)}")
            self.set_status("Baseline correction failed")

    def motion_correction(self):
        """Apply motion correction to target signals only with group-based visualization"""
        if self.reference_signal_var.get() != "410":
            messagebox.showwarning("Invalid Reference", "Motion correction requires 410nm as reference signal")
            return
            
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            motion_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "motion correction":
                    motion_tab_index = i
                    break
            
            if motion_tab_index is not None:
                self.plot_notebook.forget(motion_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
                
            target_signal = self.target_signal_var.get()
            baseline_applied = self.apply_baseline.get()

            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            motion_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(motion_tab, text="motion correction")
            
            group_notebook = ttk.Notebook(motion_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Apply motion correction and plot by group and animal
            for group_name, group_animals in groups.items():
                # Create group tab
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create notebook for animals within this group
                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Create subplots for each animal in the group
                for animal_data in group_animals:
                    animal_id = animal_data['animal_id']
                    
                    # Create tab for this animal
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)
                    
                    # Create figure and canvas for this animal
                    fig = Figure(figsize=(8, 4), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(animal_tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax1 = fig.add_subplot(2, 1, 1)
                    ax2 = fig.add_subplot(2, 1, 2)
                    
                    if animal_data.get('preprocessed_data') is None:
                        animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()
                    
                    target_columns = self.get_target_columns(animal_data, target_signal)
                    ref_columns = self.get_target_columns(animal_data, "410")

                    wavelength_colors = {
                        '410': '#0000FF',
                        '470': '#00FF00',
                        '560': '#FF0000',
                    }

                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']

                    for channel_num, target_cols, target_wavelength in target_columns:
                        if not target_cols:
                            continue
                        
                        # Find corresponding 410 reference for this channel
                        ref_cols = None
                        for ref_channel, refs, cols in ref_columns:
                            if ref_channel == channel_num and cols == '410':
                                ref_cols = refs
                                break
                        
                        if not ref_cols:
                            continue
                        
                        # Get target signal data
                        if f"CH{channel_num}_{target_wavelength}_baseline_corrected" in animal_data['preprocessed_data'].columns:
                            signal_col = f"CH{channel_num}_{target_wavelength}_baseline_corrected"
                            signal_data = animal_data['preprocessed_data'][signal_col]
                        elif f"CH{channel_num}_{target_wavelength}_smoothed" in animal_data['preprocessed_data'].columns:
                            signal_col = f"CH{channel_num}_{target_wavelength}_smoothed"
                            signal_data = animal_data['preprocessed_data'][signal_col]
                        else:
                            signal_data = animal_data['preprocessed_data'][target_cols]
                        
                        # Get 410 reference data based on baseline correction status
                        if baseline_applied:
                            # Use baseline corrected 410 if available
                            ref_410_col = f"CH{channel_num}_410_baseline_corrected"
                            if ref_410_col in animal_data['preprocessed_data'].columns:
                                ref_data = animal_data['preprocessed_data'][ref_410_col]
                            else:
                                # Fallback to smoothed or raw
                                if f"CH{channel_num}_410_smoothed" in animal_data['preprocessed_data'].columns:
                                    ref_data = animal_data['preprocessed_data'][f"CH{channel_num}_410_smoothed"]
                                else:
                                    ref_data = animal_data['preprocessed_data'][ref_cols]
                        else:
                            # Use smoothed or raw 410
                            if f"CH{channel_num}_410_smoothed" in animal_data['preprocessed_data'].columns:
                                ref_data = animal_data['preprocessed_data'][f"CH{channel_num}_410_smoothed"]
                            else:
                                ref_data = animal_data['preprocessed_data'][ref_cols]
                        
                        # Perform linear regression
                        X = ref_data.values.reshape(-1, 1)
                        y = signal_data.values
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        predicted_signal = model.predict(X)
                        
                        # Store motion corrected for each wavelength separately
                        motion_corrected_col = f"CH{channel_num}_{target_wavelength}_motion_corrected"
                        animal_data['preprocessed_data'][motion_corrected_col] = signal_data - predicted_signal
                        fitted_ref_col = f"CH{channel_num}_{target_wavelength}_fitted_ref"
                        animal_data['preprocessed_data'][fitted_ref_col] = predicted_signal
                        
                        color = wavelength_colors.get(target_wavelength, "#808080")
                        
                        # Plot original signal
                        if f"CH{channel_num}_{target_wavelength}_baseline_corrected" in animal_data['preprocessed_data'].columns:
                            ax1.plot(time_data, animal_data['preprocessed_data'][f"CH{channel_num}_{target_wavelength}_baseline_corrected"], 
                                color=color, linewidth=2, linestyle='-',
                                label=f'CH{channel_num} {target_wavelength}nm baseline corrected')
                        elif f"CH{channel_num}_{target_wavelength}_smoothed" in animal_data['preprocessed_data'].columns:
                            ax1.plot(time_data, animal_data['preprocessed_data'][f"CH{channel_num}_{target_wavelength}_smoothed"], 
                                color=color, linewidth=2, linstyle='-',
                                label=f'CH{channel_num} {target_wavelength}nm smoothed')
                        else:
                            ax1.plot(time_data, animal_data['preprocessed_data'][target_cols], 
                                color=color, linewidth=2, linestyle='-',
                                label=f'CH{channel_num} {target_wavelength}nm raw')

                        # Plot reference 410
                        ax1.plot(time_data, ref_data,    
                            color='#0000FF', linestyle='-',linewidth=2,
                            label=f'CH{channel_num} 410nm reference')
                        
                        # Plot fitted reference
                        ax1.plot(time_data, animal_data['preprocessed_data'][fitted_ref_col], 
                            color='k', linestyle='--', linewidth=1,
                            label=f'CH{channel_num} {target_wavelength}nm fitted ref')
                        
                        # Plot motion corrected
                        ax2.plot(time_data, animal_data['preprocessed_data'][motion_corrected_col], 
                            color=color, linewidth=2, linestyle='-',
                            label=f'CH{channel_num} {target_wavelength}nm motion corrected')
                    
                    reference_type = "baseline-corrected 410" if baseline_applied else "raw/smoothed 410"
                    ax1.set_title(f"Motion Correction - {animal_id} (Target: {target_signal}, Ref: {reference_type})", fontsize=14, fontweight='bold')
                    ax1.set_ylabel("Signal Intensity", fontsize=12)
                    ax1.set_xlim([time_data.min(), time_data.max()])
                    ax1.tick_params(axis='both', labelsize=10)
                    ax1.grid(False)
                    ax1.legend()
                    ax2.set_xlabel("Time (s)", fontsize=12)
                    ax2.set_ylabel("Signal Intensity", fontsize=12)
                    ax2.set_xlim([time_data.min(), time_data.max()])
                    ax2.tick_params(axis='both', labelsize=10)
                    ax2.grid(False)
                    ax2.legend()
                    
                    # Add event markers if available
                    if 'event_data' in animal_data:
                        event_data = animal_data['event_data']
                        event_time_absolute = animal_data.get('event_time_absolute', False)
                        if event_time_absolute:
                            start_col = 'start_absolute'
                            end_col = 'end_absolute'
                        else:
                            start_col = 'start_time'
                            end_col = 'end_time'
                        
                        video_start = animal_data['video_start_fiber']
                        
                        for _, row in event_data.iterrows():
                            if row['Event Type'] == 0:
                                color = "#ffffff"
                            elif row['Event Type'] == 3:
                                color = "#ffffff"
                            else:
                                color = "#fc0000"

                            start_time = row[start_col] - video_start
                            end_time = row[end_col] - video_start
                            ax1.axvspan(start_time, end_time, color=color, alpha=0.5)
                            ax2.axvspan(start_time, end_time, color=color, alpha=0.5)

                    canvas.draw()
                
            self.set_status(f"Motion correction applied (target: {target_signal}, ref: {reference_type})")
            messagebox.showinfo("Success", f"Motion correction applied to {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Motion correction failed: {str(e)}")
            self.set_status("Motion correction failed")

    def apply_preprocessing(self):
        """Apply preprocessing to selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            if self.apply_smooth.get():
                self.smooth_data()
            if self.apply_baseline.get():
                self.baseline_correction()
            if self.apply_motion.get() and self.reference_signal_var.get() == "410":
                self.motion_correction()
            
            messagebox.showinfo("Success", f"Preprocessing applied to {len(selected_indices)} animals")
            self.set_status(f"Preprocessing applied to {len(selected_indices)} animals")
                
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.set_status("Preprocessing failed")

    def calculate_and_plot_dff(self):
        """Calculate dF/F for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            target_signal = self.target_signal_var.get()
            ref_wavelength = self.reference_signal_var.get()

            dff_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "dF/F":
                    dff_tab_index = i
                    break
            
            if dff_tab_index is not None:
                self.plot_notebook.forget(dff_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
                
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            dff_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(dff_tab, text="dF/F")
            
            group_notebook = ttk.Notebook(dff_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Calculate dF/F and plot by group and animal
            for group_name, group_animals in groups.items():
                # Create group tab
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create notebook for animals within this group
                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Create subplots for each animal in the group
                for animal_data in group_animals:
                    animal_id = animal_data['animal_id']
                    
                    # Create tab for this animal
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)
                    
                    # Create figure and canvas for this animal
                    fig = Figure(figsize=(8, 4), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(animal_tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax = fig.add_subplot(111)
                    
                    if animal_data.get('preprocessed_data') is None:
                        animal_data['preprocessed_data'] = animal_data['fiber_cropped'].copy()

                    animal_data['dff_data'] = {}
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
                    
                    if not any(baseline_mask):
                        continue
                        
                    target_columns = self.get_target_columns(animal_data, target_signal)

                    for channel_num, cols, wavelength in target_columns:
                        if not cols:
                            continue
                        
                        if f"CH{channel_num}_{wavelength}_smoothed" in animal_data['preprocessed_data'].columns:
                            raw_target = animal_data['preprocessed_data'][f"CH{channel_num}_{wavelength}_smoothed"]
                        else:
                            raw_target = animal_data['preprocessed_data'][cols]
                        
                        median_full = np.median(raw_target)
                        
                        if ref_wavelength == "410" and self.apply_baseline.get():
                            if f"CH{channel_num}_{wavelength}_motion_corrected" in animal_data['preprocessed_data'].columns:
                                motion_corrected = animal_data['preprocessed_data'][f"CH{channel_num}_{wavelength}_motion_corrected"]
                                dff_data = motion_corrected / median_full
                                print(f"ref_wavelength: {ref_wavelength}, apply_baseline: {self.apply_baseline.get()}")
                            else:
                                continue
                        elif ref_wavelength == "410" and not self.apply_baseline.get():
                            if f"CH{channel_num}_{wavelength}_fitted_ref" in animal_data['preprocessed_data'].columns:
                                fitted_ref = animal_data['preprocessed_data'][f"CH{channel_num}_{wavelength}_fitted_ref"]
                                dff_data = (raw_target - fitted_ref) / fitted_ref
                                print(f"ref_wavelength: {ref_wavelength}, apply_baseline: {self.apply_baseline.get()}")
                            else:
                                continue
                        elif ref_wavelength == "baseline" and self.apply_baseline.get():
                            if f"CH{channel_num}_{wavelength}_baseline_pred" in animal_data['preprocessed_data'].columns:
                                baseline_fitted = animal_data['preprocessed_data'][f"CH{channel_num}_{wavelength}_baseline_pred"]
                                baseline_median = np.median(raw_target[baseline_mask])
                                dff_data = (raw_target - baseline_fitted) / baseline_median
                                print(f"ref_wavelength: {ref_wavelength}, apply_baseline: {self.apply_baseline.get()}")
                            else:
                                continue
                        elif ref_wavelength == "baseline" and not self.apply_baseline.get():
                            baseline_median = np.median(raw_target[baseline_mask])
                            dff_data = (raw_target - baseline_median) / baseline_median
                            print(f"ref_wavelength: {ref_wavelength}, apply_baseline: {self.apply_baseline.get()}")
                        else:
                            dff_data = (raw_target - median_full) / median_full
                        
                        animal_data['dff_data'][f"{channel_num}_{wavelength}"] = dff_data
                        
                        wavelength_colors = {
                            '410': '#0000FF',
                            '470': '#00FF00',
                            '560': '#FF0000',
                        }
                        color = wavelength_colors.get(wavelength, "#808080")
                        
                        ax.plot(time_data, dff_data, 
                            color=color, linewidth=2, linestyle='-',
                            label=f'CH{channel_num} {wavelength} nm dF/F')
                    
                    ax.set_title(f"ΔF/F - {animal_id} (Target: {target_signal})", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Time (s)", fontsize=12)
                    ax.set_ylabel("ΔF/F", fontsize=12)
                    ax.set_xlim([time_data.min(),time_data.max()])
                    ax.tick_params(axis='both', labelsize=10)
                    ax.grid(False)
                    ax.legend()
                    
                    # Add event markers if available
                    if 'event_data' in animal_data:
                        event_data = animal_data['event_data']
                        event_time_absolute = animal_data.get('event_time_absolute', False)
                        if event_time_absolute:
                            start_col = 'start_absolute'
                            end_col = 'end_absolute'
                        else:
                            start_col = 'start_time'
                            end_col = 'end_time'
                        
                        video_start = animal_data['video_start_fiber']
                        
                        for _, row in event_data.iterrows():
                            if row['Event Type'] == 0:
                                color = "#ffffff"
                            elif row['Event Type'] == 3:
                                color = "#ffffff"
                            else:
                                color = "#fc0000"

                            start_time = row[start_col] - video_start
                            end_time = row[end_col] - video_start
                            ax.axvspan(start_time, end_time, color=color, alpha=0.5)

                    canvas.draw()
                
            self.set_status(f"dF/F calculated and plotted (target: {target_signal})")
            messagebox.showinfo("Success", f"dF/F calculated for {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"dF/F calculation failed: {str(e)}")
            self.set_status("dF/F calculation failed")

    def calculate_and_plot_zscore(self):
        """Calculate Z-score for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            zscore_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Z-score":
                    zscore_tab_index = i
                    break
            
            if zscore_tab_index is not None:
                self.plot_notebook.forget(zscore_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
                
            target_signal = self.target_signal_var.get()

            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            zscore_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(zscore_tab, text="Z-score")
            
            group_notebook = ttk.Notebook(zscore_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Calculate Z-score and plot by group and animal
            for group_name, group_animals in groups.items():
                # Create group tab
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create notebook for animals within this group
                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Create subplots for each animal in the group
                for animal_data in group_animals:
                    animal_id = animal_data['animal_id']
                    
                    # Create tab for this animal
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)
                    
                    # Create figure and canvas for this animal
                    fig = Figure(figsize=(8, 4), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(animal_tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    ax = fig.add_subplot(111)
                    
                    animal_data['zscore_data'] = {}
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    baseline_mask = (time_data >= self.baseline_period[0]) & (time_data <= self.baseline_period[1])
                    
                    if not any(baseline_mask):
                        continue
                        
                    target_columns = self.get_target_columns(animal_data, target_signal)

                    for channel_num, cols, wavelength in target_columns:
                        if not cols:
                            continue
                        
                        dff_key = f"{channel_num}_{wavelength}"
                        dff_data = animal_data['dff_data'].get(dff_key)
                        if dff_data is None:
                            continue
                        
                        baseline_values = dff_data[baseline_mask]
                        baseline_mean = np.mean(baseline_values)
                        baseline_std = np.std(baseline_values)
                        
                        if baseline_std > 0:
                            zscore_data = (dff_data - baseline_mean) / baseline_std
                            animal_data['zscore_data'][dff_key] = zscore_data
                            
                            colors = ["#00FF00", "#0AC400FF", "#00940F", "#0D6000", "#073B00"]
                            color_idx = (channel_num - 1) % len(colors)
                            color = colors[color_idx]
                            
                            ax.plot(time_data, zscore_data, 
                                color=color, linewidth=2, linestyle='-',
                                label=f'CH{channel_num} {wavelength} nm Z-score')
                    
                    ax.set_title(f"Z-score - {animal_id} (Target: {target_signal})", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Time (s)", fontsize=12)
                    ax.set_ylabel("Z-score", fontsize=12)
                    ax.set_xlim([time_data.min(), time_data.max()])
                    ax.tick_params(axis='both', labelsize=10)
                    ax.grid(False)
                    ax.legend()
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                    # Add event markers if available
                    if 'event_data' in animal_data:
                        event_data = animal_data['event_data']
                        event_time_absolute = animal_data.get('event_time_absolute', False)
                        if event_time_absolute:
                            start_col = 'start_absolute'
                            end_col = 'end_absolute'
                        else:
                            start_col = 'start_time'
                            end_col = 'end_time'
                        
                        video_start = animal_data['video_start_fiber']
                        
                        for _, row in event_data.iterrows():
                            if row['Event Type'] == 0:
                                color = "#ffffff"
                            elif row['Event Type'] == 3:
                                color = "#ffffff"
                            else:
                                color = "#fc0000"

                            start_time = row[start_col] - video_start
                            end_time = row[end_col] - video_start
                            ax.axvspan(start_time, end_time, color=color, alpha=0.5)

                    canvas.draw()
                
            self.set_status(f"Z-score calculated and plotted (target: {target_signal})")
            messagebox.showinfo("Success", f"Z-score calculated for {len(selected_indices)} animals")
        except Exception as e:
            messagebox.showerror("Error", f"Z-score calculation failed: {str(e)}")
            self.set_status("Z-score calculation failed")
            
    def compute_event_zscore_from_dff(self, dff_data, time_data, event_start, event_end, pre_window, post_window, post_flag):
        """Compute event-related z-score from dF/F data"""
        event_window_start = event_start - pre_window
        if post_flag is True:
            event_window_end = event_end + post_window
        else:
            event_window_end = event_start + post_window
        
        time_data = np.asarray(time_data, dtype=float)
        dff_data = np.asarray(dff_data, dtype=float)

        window_mask = (time_data >= event_window_start) & (time_data <= event_window_end)
        window_time = time_data[window_mask]
        window_dff = dff_data[window_mask]
        
        valid_mask = ~np.isnan(window_dff) & ~np.isnan(window_time)
        window_time = window_time[valid_mask]
        window_dff = window_dff[valid_mask]

        if len(window_dff) == 0:
            return np.array([]), np.array([]), None
        
        baseline_mask = (window_time >= event_window_start) & (window_time < event_start)
        baseline_dff = window_dff[baseline_mask]
        
        if len(baseline_dff) < 2:
            return window_time - event_start, np.full(len(window_dff), np.nan), None
        
        baseline_mean = np.mean(baseline_dff)
        baseline_std = np.std(baseline_dff)
        
        if baseline_std == 0:
            baseline_std = 1e-6
        
        zscores = (window_dff - baseline_mean) / baseline_std
        
        return window_time - event_start, zscores, (baseline_mean, baseline_std)

    def compute_event_dff_from_dff(self, dff_data, time_data, event_start, event_end, pre_window, post_window, post_flag):
        """Compute event-related dF/F from dF/F data"""
        event_window_start = event_start - pre_window
        if post_flag is True:
            event_window_end = event_end + post_window
        else:
            event_window_end = event_start + post_window
        
        time_data = np.asarray(time_data, dtype=float)
        dff_data = np.asarray(dff_data, dtype=float)

        window_mask = (time_data >= event_window_start) & (time_data <= event_window_end)
        window_time = time_data[window_mask]
        window_dff = dff_data[window_mask]
        
        valid_mask = ~np.isnan(window_dff) & ~np.isnan(window_time)
        window_time = window_time[valid_mask]
        window_dff = window_dff[valid_mask]

        if len(window_dff) == 0:
            return np.array([]), np.array([])
        
        return window_time - event_start, window_dff
    
    def show_event_activity_dialog(self):
        """Show dialog for event activity and heatmap parameters with group-based buttons"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Group animals by group
        groups = {}
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            group = animal_data.get('group', 'Unknown')
            if group not in groups:
                groups[group] = []
            groups[group].append(animal_data)
        
        # Get available event types from selected animals by group
        group_event_types = {}
        for group_name, group_animals in groups.items():
            event_types = set()
            for animal_data in group_animals:
                if 'event_data' in animal_data:
                    types = animal_data['event_data']['Event Type'].unique()
                    event_types.update([t for t in types if t in [1, 2]])
            group_event_types[group_name] = sorted(event_types)
        
        # Filter out groups with no valid event types
        valid_groups = {group: events for group, events in group_event_types.items() if events}
        
        if not valid_groups:
            messagebox.showwarning("No Events", "No valid event types found (only type 1 and 2 are supported)")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Event-Activity & Heatmap Settings")
        dialog.geometry("300x690")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Create notebook for group tabs
        group_notebook = ttk.Notebook(main_frame)
        group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        group_vars = {}  # Store variables for each group
        
        def update_event_buttons(group_name, event_type_var, scrollable_frame):
            """Update event buttons when event type changes"""
            # Clear existing buttons
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            
            # Get all events for this group across all animals
            all_events_info = []
            for animal_data in groups[group_name]:
                if 'event_data' in animal_data:
                    event_data = animal_data['event_data']
                    events = event_data[event_data['Event Type'] == int(event_type_var.get().split()[0])]
                    for event_idx, _ in enumerate(events.iterrows()):
                        event_name = "Sound" if int(event_type_var.get().split()[0]) == 1 else "Shock"
                        event_label = f"{event_name}{event_idx+1}"
                        all_events_info.append(event_label)
            
            # Get intersection of events across all animals
            event_counts = {}
            for event_label in all_events_info:
                event_counts[event_label] = event_counts.get(event_label, 0) + 1
            
            total_animals = len(groups[group_name])
            common_events = [event_label for event_label, count in event_counts.items() if count == total_animals]
            
            if not common_events:
                messagebox.showwarning("No Common Events", 
                                    f"No common events found across all animals in group {group_name}")
                return []
            
            # Create buttons for each common event
            event_buttons = {}
            for event_label in common_events:
                var = tk.BooleanVar(value=True)  # Default selected
                event_buttons[event_label] = var
                
                btn = ttk.Checkbutton(
                    scrollable_frame,
                    text=event_label,
                    variable=var
                )
                btn.pack(anchor="w", padx=5, pady=2)
            
            return event_buttons
        
        for group_name, event_types in valid_groups.items():
            # Create tab for this group
            group_tab = ttk.Frame(group_notebook)
            group_notebook.add(group_tab, text=group_name)
            
            # Event type selection
            ttk.Label(group_tab, text="Event Type:").pack(anchor="w", pady=5)
            event_type_var = tk.StringVar(value=str(min(event_types)))
            
            # Create mapping for event types
            event_type_mapping = {1: "Sound", 2: "Shock"}
            event_menu_values = [f"{event_type} ({event_type_mapping[event_type]})" for event_type in sorted(event_types)]
            event_menu = ttk.OptionMenu(group_tab, event_type_var, event_menu_values[0], *event_menu_values)
            event_menu.pack(fill="x", pady=5)
            
            # Event selection button frame
            ttk.Label(group_tab, text="Select Events:").pack(anchor="w", pady=5)
            
            # Create scrollable frame for event buttons
            button_frame_container = ttk.Frame(group_tab, width=200)
            button_frame_container.pack(fill="both", expand=False, pady=5)
            
            canvas = tk.Canvas(button_frame_container, height=100, width=200)
            scrollbar = ttk.Scrollbar(button_frame_container, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Initial event buttons
            event_buttons = update_event_buttons(group_name, event_type_var, scrollable_frame)
            
            # Store variables for this group
            group_vars[group_name] = {
                'event_type_var': event_type_var,
                'event_buttons': event_buttons,
                'scrollable_frame': scrollable_frame
            }
            
            # Update event buttons when event type changes
            def on_event_type_change(var, group=group_name):
                event_type = int(var.get().split(' ')[0])  # Extract the number from "1 (sound)"
                group_vars[group]['event_buttons'] = update_event_buttons(
                    group, 
                    tk.StringVar(value=str(event_type)), 
                    group_vars[group]['scrollable_frame']
                )
            
            event_type_var.trace_add("write", lambda *args, var=event_type_var: on_event_type_change(var))
        
        # Time windows frame (common for all groups)
        time_frame = ttk.LabelFrame(main_frame, text="Time Settings")
        time_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(time_frame, text="Pre-event Window (s):").grid(row=0, column=0, sticky="w", pady=5)
        pre_entry = ttk.Entry(time_frame, width=10)
        pre_entry.insert(0, "10")
        pre_entry.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(time_frame, text="Post-event Window (s):").grid(row=1, column=0, sticky="w", pady=5)
        post_entry = ttk.Entry(time_frame, width=10)
        post_entry.insert(0, "20")
        post_entry.grid(row=1, column=1, sticky="w", padx=5)
        
        # Curve spacing setting
        spacing_frame = ttk.LabelFrame(main_frame, text="Curve Spacing")
        spacing_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(spacing_frame, text="Vertical Spacing:").grid(row=0, column=0, sticky="w", pady=5)
        spacing_var = tk.DoubleVar(value=5.0)
        spacing_scale = ttk.Scale(spacing_frame, from_=0.0, to=20.0, orient=tk.HORIZONTAL, 
                                variable=spacing_var, length=100)
        spacing_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        spacing_value = ttk.Label(spacing_frame, textvariable=spacing_var)
        spacing_value.grid(row=0, column=2, padx=5)
        
        # Smoothing options
        smooth_frame = ttk.LabelFrame(main_frame, text="Smoothing Options")
        smooth_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(smooth_frame, text="Smoothing Method:").grid(row=0, column=0, sticky="w", pady=5)
        smooth_var = tk.StringVar(value="None")
        smooth_menu = ttk.OptionMenu(smooth_frame, smooth_var, "None", "None", "Moving Average", "Savitzky-Golay")
        smooth_menu.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(smooth_frame, text="Smooth Window:").grid(row=1, column=0, sticky="w", pady=5)
        smooth_window = ttk.Entry(smooth_frame, width=10)
        smooth_window.insert(0, "11")
        smooth_window.grid(row=1, column=1, sticky="w", padx=5)
        
        # DownSample options
        downsample_frame = ttk.LabelFrame(main_frame, text="DownSample Options")
        downsample_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(downsample_frame, text="DownSmaple Rate:").grid(row=1, column=0, sticky="w", pady=5)
        downsample_rate = ttk.Entry(downsample_frame, width=10)
        downsample_rate.insert(0, "1")
        downsample_rate.grid(row=1, column=1, sticky="w", padx=5)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        
        def select_all_events():
            """Select all events in current group tab"""
            current_tab = group_notebook.tab(group_notebook.select(), "text")
            for var in group_vars[current_tab]['event_buttons'].values():
                var.set(True)
        
        def deselect_all_events():
            """Deselect all events in current group tab"""
            current_tab = group_notebook.tab(group_notebook.select(), "text")
            for var in group_vars[current_tab]['event_buttons'].values():
                var.set(False)

        def apply():
            try:
                all_group_params = {}
                
                for group_name in valid_groups.keys():
                    if group_name not in group_vars:
                        continue
                        
                    # Get selected events for this group
                    selected_events = []
                    for event_label, var in group_vars[group_name]['event_buttons'].items():
                        if var.get():
                            selected_events.append(event_label)
                    
                    if not selected_events:
                        messagebox.showwarning("No Selection", f"Please select at least one event for group {group_name}")
                        return
                    
                    all_group_params[group_name] = {
                        'event_type': int(group_vars[group_name]['event_type_var'].get().split(' ')[0]),  # Extract number
                        'selected_events': selected_events,
                        'pre_window': float(pre_entry.get()),
                        'post_window': float(post_entry.get()),
                        'smooth_method': smooth_var.get(),
                        'smooth_window': int(smooth_window.get()) if smooth_var.get() != "None" else None,
                        'curve_spacing': spacing_var.get(),
                        'downsample_rate': float(downsample_rate.get())
                    }
                
                self.plot_event_activity_and_heatmap(all_group_params, preview=False)
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
        
        ttk.Button(btn_frame, text="Select All",   command=select_all_events).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(btn_frame, text="Deselect All", command=deselect_all_events).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(btn_frame, text="Apply",        command=apply).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button(btn_frame, text="Close",        command=dialog.destroy).grid(row=1, column=1, padx=5, pady=2)

        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

    def show_experiment_activity_dialog(self):
        """Show dialog for experiment activity parameters with group-based buttons"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Group animals by group
        groups = {}
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            group = animal_data.get('group', 'Unknown')
            if group not in groups:
                groups[group] = []
            groups[group].append(animal_data)
        
        # Get available event types from selected animals by group
        group_event_types = {}
        for group_name, group_animals in groups.items():
            event_types = set()
            for animal_data in group_animals:
                if 'event_data' in animal_data:
                    types = animal_data['event_data']['Event Type'].unique()
                    event_types.update([t for t in types if t in [1, 2]])
            group_event_types[group_name] = sorted(event_types)
        
        # Filter out groups with no valid event types
        valid_groups = {group: events for group, events in group_event_types.items() if events}
        
        if not valid_groups:
            messagebox.showwarning("No Events", "No valid event types found (only type 1 and 2 are supported)")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Experiment-Activity Settings")
        dialog.geometry("300x620")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Create notebook for group tabs
        group_notebook = ttk.Notebook(main_frame)
        group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        group_vars = {}  # Store variables for each group
        
        def update_event_buttons(group_name, event_type_var, scrollable_frame):
            """Update event buttons when event type changes"""
            # Clear existing buttons
            for widget in scrollable_frame.winfo_children():
                widget.destroy()
            
            # Get all events for this group across all animals
            all_events_info = []
            for animal_data in groups[group_name]:
                if 'event_data' in animal_data:
                    event_data = animal_data['event_data']
                    events = event_data[event_data['Event Type'] == int(event_type_var.get().split()[0])]
                    for event_idx, _ in enumerate(events.iterrows()):
                        event_name = "Sound" if int(event_type_var.get().split()[0]) == 1 else "Shock"
                        event_label = f"{event_name}{event_idx+1}"
                        all_events_info.append(event_label)
            
            # Get intersection of events across all animals
            event_counts = {}
            for event_label in all_events_info:
                event_counts[event_label] = event_counts.get(event_label, 0) + 1
            
            total_animals = len(groups[group_name])
            common_events = [event_label for event_label, count in event_counts.items() if count == total_animals]
            
            if not common_events:
                messagebox.showwarning("No Common Events", 
                                    f"No common events found across all animals in group {group_name}")
                return []
            
            # Create buttons for each common event
            event_buttons = {}
            for event_label in common_events:
                var = tk.BooleanVar(value=True)  # Default selected
                event_buttons[event_label] = var
                
                btn = ttk.Checkbutton(
                    scrollable_frame,
                    text=event_label,
                    variable=var
                )
                btn.pack(anchor="w", padx=5, pady=2)
            
            return event_buttons
        
        for group_name, event_types in valid_groups.items():
            # Create tab for this group
            group_tab = ttk.Frame(group_notebook)
            group_notebook.add(group_tab, text=group_name)
            
            # Event type selection
            ttk.Label(group_tab, text="Event Type:").pack(anchor="w", pady=5)
            event_type_var = tk.StringVar(value=str(min(event_types)))
            
            # Create mapping for event types
            event_type_mapping = {1: "Sound", 2: "Shock"}
            event_menu_values = [f"{event_type} ({event_type_mapping[event_type]})" for event_type in sorted(event_types)]
            event_menu = ttk.OptionMenu(group_tab, event_type_var, event_menu_values[0], *event_menu_values)
            event_menu.pack(fill="x", pady=5)
            
            # Event selection button frame
            ttk.Label(group_tab, text="Select Events:").pack(anchor="w", pady=5)
            
            # Create scrollable frame for event buttons
            button_frame_container = ttk.Frame(group_tab, width=200)
            button_frame_container.pack(fill="both", expand=False, pady=5)
            
            canvas = tk.Canvas(button_frame_container, height=100, width=200)
            scrollbar = ttk.Scrollbar(button_frame_container, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Initial event buttons
            event_buttons = update_event_buttons(group_name, event_type_var, scrollable_frame)
            
            # Store variables for this group
            group_vars[group_name] = {
                'event_type_var': event_type_var,
                'event_buttons': event_buttons,
                'scrollable_frame': scrollable_frame
            }
            
            # Update event buttons when event type changes
            def on_event_type_change(var, group=group_name):
                event_type = int(var.get().split(' ')[0])  # Extract the number from "1 (sound)"
                group_vars[group]['event_buttons'] = update_event_buttons(
                    group, 
                    tk.StringVar(value=str(event_type)), 
                    group_vars[group]['scrollable_frame']
                )
            
            event_type_var.trace_add("write", lambda *args, var=event_type_var: on_event_type_change(var))
        
        # Time windows frame (common for all groups)
        time_frame = ttk.LabelFrame(main_frame, text="Time Settings")
        time_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(time_frame, text="Pre-event Window (s):").grid(row=0, column=0, sticky="w", pady=5)
        pre_entry = ttk.Entry(time_frame, width=10)
        pre_entry.insert(0, "10")
        pre_entry.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(time_frame, text="Post-event Window (s):").grid(row=1, column=0, sticky="w", pady=5)
        post_entry = ttk.Entry(time_frame, width=10)
        post_entry.insert(0, "20")
        post_entry.grid(row=1, column=1, sticky="w", padx=5)
        
        # Smoothing options
        smooth_frame = ttk.LabelFrame(main_frame, text="Smoothing Options")
        smooth_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(smooth_frame, text="Smoothing Method:").grid(row=0, column=0, sticky="w", pady=5)
        smooth_var = tk.StringVar(value="None")
        smooth_menu = ttk.OptionMenu(smooth_frame, smooth_var, "None", "None", "Moving Average", "Savitzky-Golay")
        smooth_menu.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(smooth_frame, text="Smooth Window:").grid(row=1, column=0, sticky="w", pady=5)
        smooth_window = ttk.Entry(smooth_frame, width=10)
        smooth_window.insert(0, "11")
        smooth_window.grid(row=1, column=1, sticky="w", padx=5)

        # DownSample options
        downsample_frame = ttk.LabelFrame(main_frame, text="DownSample Options")
        downsample_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(downsample_frame, text="DownSmaple Rate:").grid(row=1, column=0, sticky="w", pady=5)
        downsample_rate = ttk.Entry(downsample_frame, width=10)
        downsample_rate.insert(0, "1")
        downsample_rate.grid(row=1, column=1, sticky="w", padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        
        def select_all_events():
            """Select all events in current group tab"""
            current_tab = group_notebook.tab(group_notebook.select(), "text")
            for var in group_vars[current_tab]['event_buttons'].values():
                var.set(True)
        
        def deselect_all_events():
            """Deselect all events in current group tab"""
            current_tab = group_notebook.tab(group_notebook.select(), "text")
            for var in group_vars[current_tab]['event_buttons'].values():
                var.set(False)

        def apply():
            try:
                all_group_params = {}
                
                for group_name in valid_groups.keys():
                    if group_name not in group_vars:
                        continue
                        
                    # Get selected events for this group
                    selected_events = []
                    for event_label, var in group_vars[group_name]['event_buttons'].items():
                        if var.get():
                            selected_events.append(event_label)
                    
                    if not selected_events:
                        messagebox.showwarning("No Selection", f"Please select at least one event for group {group_name}")
                        return
                    
                    all_group_params[group_name] = {
                        'event_type': int(group_vars[group_name]['event_type_var'].get().split(' ')[0]),  # Extract number
                        'selected_events': selected_events,
                        'pre_window': float(pre_entry.get()),
                        'post_window': float(post_entry.get()),
                        'smooth_method': smooth_var.get(),
                        'smooth_window': int(smooth_window.get()) if smooth_var.get() != "None" else None,
                        'downsample_rate': float(downsample_rate.get())
                    }
                
                self.plot_experiment_activity(all_group_params, preview=False)
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))
        
        ttk.Button(btn_frame, text="Select All",   command=select_all_events).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(btn_frame, text="Deselect All", command=deselect_all_events).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(btn_frame, text="Apply",        command=apply).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button(btn_frame, text="Close",        command=dialog.destroy).grid(row=1, column=1, padx=5, pady=2)

        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

    def plot_event_activity_and_heatmap(self, all_group_params, preview=False):
        """Plot event-related activity and heatmap - each wavelength separately"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
        
        try:
            # Clear existing tabs
            event_activity_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "event activity":
                    event_activity_tab_index = i
                    break
            
            if event_activity_tab_index is not None:
                self.plot_notebook.forget(event_activity_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
            
            event_activity_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(event_activity_tab, text="event activity")
            
            target_signal = self.target_signal_var.get()

            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            group_notebook = ttk.Notebook(event_activity_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)

            # Define wavelength colors
            wavelength_colors = {
                '410': '#0000FF',
                '470': '#00FF00',
                '560': '#FF0000',
            }

            # Process each group that has parameters
            for group_name, group_params in all_group_params.items():
                if group_name not in groups:
                    continue
                    
                group_animals = groups[group_name]
                
                # Create tab for this group
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create figure with 2 subplots
                fig = Figure(figsize=(10, 8), dpi=100)
                canvas = FigureCanvasTkAgg(fig, master=group_tab)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                
                toolbar_frame = ttk.Frame(group_tab)
                toolbar_frame.pack(fill="x")
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                
                # Create subplots
                ax1 = fig.add_subplot(2, 1, 1)  # Event-related activity
                ax2 = fig.add_subplot(2, 1, 2)  # Heatmap
                
                # Collect data for this group - NOW SEPARATED BY WAVELENGTH
                all_event_responses = {}  # key: (event_label, wavelength), value: list of (time, zscore)
                heatmap_data = []
                heatmap_y_labels = []
                event_boundaries = []
                
                event_type = group_params['event_type']
                event_name = "Sound" if event_type == 1 else "Shock"
                
                # Get event indices for selected events
                selected_event_indices = {}
                for animal_data in group_animals:
                    if 'event_data' in animal_data:
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                        for event_idx, (_, event) in enumerate(events.iterrows()):
                            event_label = f"{event_name}{event_idx+1}"
                            if event_label in group_params['selected_events']:
                                selected_event_indices[event_label] = event_idx
                
                # Sort events by label
                sorted_event_labels = sorted(selected_event_indices.keys(), 
                                        key=lambda x: int(x.replace(event_name, "")))
                
                # Process each selected event
                for event_label in sorted_event_labels:
                    event_idx = selected_event_indices[event_label]
                    
                    # Collect all responses for this event - SEPARATED BY WAVELENGTH
                    for animal_data in group_animals:
                        if 'dff_data' not in animal_data or 'event_data' not in animal_data:
                            continue
                        
                        # Get event data
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                        if event_idx >= len(events):
                            continue
                        
                        event = events.iloc[event_idx]
                        
                        time_col = animal_data['channels']['time']
                        time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                        
                        if animal_data.get('event_time_absolute', False):
                            start_time = event['start_absolute'] - animal_data['video_start_fiber']
                            end_time = event['end_absolute'] - animal_data['video_start_fiber']
                        else:
                            start_time = event['start_time']
                            end_time = event['end_time']
                        
                        duration = end_time - start_time

                        target_columns = self.get_target_columns(animal_data, target_signal)
                        
                        # Collect responses for each wavelength SEPARATELY
                        for channel_num, cols, wavelength in target_columns:
                            if not cols:
                                continue
                            
                            dff_key = f"{channel_num}_{wavelength}"
                            dff_data = animal_data['dff_data'].get(dff_key)
                            if dff_data is None:
                                continue
                            
                            event_time_rel, event_zscore, _ = self.compute_event_zscore_from_dff(
                                dff_data, time_data, start_time, end_time,
                                group_params['pre_window'], group_params['post_window'], post_flag=True
                            )
                            
                            if len(event_time_rel) > 0:
                                if group_params['smooth_method'] == "Moving Average":
                                    kernel = np.ones(group_params['smooth_window']) / group_params['smooth_window']
                                    event_zscore = np.convolve(event_zscore, kernel, mode='same')
                                elif group_params['smooth_method'] == "Savitzky-Golay":
                                    event_zscore = savgol_filter(event_zscore, group_params['smooth_window'], 3)
                                
                                # Store by (event_label, wavelength) key
                                key = (event_label, wavelength)
                                if key not in all_event_responses:
                                    all_event_responses[key] = []
                                all_event_responses[key].append((event_time_rel, event_zscore))
                
                # Plot event-related activity (mean ± SEM for each event and wavelength)
                curve_spacing = group_params.get('curve_spacing', 5.0)
                
                # Calculate common time using fps_fiber
                if all_event_responses:
                    all_time_data = []
                    for responses in all_event_responses.values():
                        for time_rel, _ in responses:
                            all_time_data.extend(time_rel)
                    
                    if all_time_data:
                        max_time = max(all_time_data)
                        min_time = min(all_time_data)
                        common_time = np.linspace(min_time, max_time, int((max_time - min_time) * fps_fiber / group_params['downsample_rate']))
                    else:
                        common_time = np.linspace(-group_params['pre_window'], group_params['post_window'], 100)
                else:
                    common_time = np.linspace(-group_params['pre_window'], group_params['post_window'], 100)
                
                # Plot each event-wavelength combination
                plot_idx = 0
                cmap = get_cmap('viridis')
                norm = plt.Normalize(0, len(sorted_event_labels)-1)
                for event_label in sorted_event_labels:
                    # Get all wavelengths for this event
                    wavelengths_in_event = set()
                    for key in all_event_responses.keys():
                        if key[0] == event_label:
                            wavelengths_in_event.add(key[1])
                    
                    for wavelength in sorted(wavelengths_in_event):
                        key = (event_label, wavelength)
                        if key not in all_event_responses:
                            continue
                        
                        responses = all_event_responses[key]
                        
                        # Interpolate to common time axis
                        common_responses = []
                        for time_rel, zscore in responses:
                            if len(time_rel) > 0 and len(zscore) == len(time_rel):
                                interp_zscore = np.interp(common_time, time_rel, zscore)
                                common_responses.append(interp_zscore)
                        
                        if not common_responses:
                            continue
                        
                        # Calculate mean and SEM
                        mean_response = np.nanmean(common_responses, axis=0)
                        sem_response = np.nanstd(common_responses, axis=0, ddof=1) / np.sqrt(len(common_responses))
                        
                        # Apply vertical spacing
                        vertical_offset = plot_idx * curve_spacing
                        
                        # Get color for this wavelength
                        color = wavelength_colors.get(wavelength, "#808080")
                        
                        ax1.plot(common_time, mean_response + vertical_offset, 
                                color=cmap(norm(plot_idx)), linewidth=2, label=f"{event_label} {wavelength}nm")
                        ax1.fill_between(common_time, 
                                    mean_response + vertical_offset - sem_response, 
                                    mean_response + vertical_offset + sem_response, 
                                    color=color, alpha=0.3)
                        ax1.axhline(0 + vertical_offset, color='#888888', linestyle='--', linewidth=1)
                        
                        plot_idx += 1
                
                if sorted_event_labels:
                    ax1.set_xlim(common_time.min(), common_time.max())
                    
                    # Mark event positions
                    for animal_data in group_animals:
                        if 'event_data' in animal_data:
                            events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                            events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                            for _, event2 in events2.iterrows():
                                if animal_data.get('event_time_absolute', False):
                                    start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                                    end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    start2 = event2['start_time']
                                    end2 = event2['end_time']
                                
                                for _, event1 in events.iterrows():
                                    if animal_data.get('event_time_absolute', False):
                                        event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                                        event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                                    else:
                                        event1_start = event1['start_time']
                                        event1_end = event1['end_time']
                                    
                                    if start2 >= event1_start and end2 <= event1_end:
                                        rel_start = start2 - event1_start
                                        ax1.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1,
                                                label='Shock Start' if 'Shock Start' not in ax1.get_legend_handles_labels()[1] else "")
                    
                    ax1.axvline(0, color='#000000', linestyle='--', linewidth=1, label=f"{event_name} Start")
                    ax1.axvline(duration, color='#000000', linestyle='--', linewidth=1, label=f'{event_name} End')      
                    # ax1.set_xlabel("Time (s)", fontsize=12)
                    ax1.set_ylabel("Z-Score", fontsize=12)
                    ax1.tick_params(axis='both', labelsize=10)
                    ax1.set_title(f"Event-Related Activity - Group {group_name} ({event_name}, Target: {target_signal})", fontsize=14, fontweight='bold')
                    ax1.grid(False)
                    ax1.legend()
                
                # Prepare heatmap data - each wavelength gets its own row
                y_position = 0
                
                for event_label in sorted_event_labels:
                    event_idx = selected_event_indices[event_label]
                    
                    # For each animal and wavelength, add to heatmap
                    for animal_data in group_animals:
                        if 'dff_data' not in animal_data or 'event_data' not in animal_data:
                            continue
                        
                        # Get event data
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                        if event_idx >= len(events):
                            continue
                        
                        event = events.iloc[event_idx]
                        
                        time_col = animal_data['channels']['time']
                        time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                        
                        if animal_data.get('event_time_absolute', False):
                            start_time = event['start_absolute'] - animal_data['video_start_fiber']
                            end_time = event['end_absolute'] - animal_data['video_start_fiber']
                        else:
                            start_time = event['start_time']
                            end_time = event['end_time']
                        
                        duration = end_time - start_time
                        
                        target_columns = self.get_target_columns(animal_data, target_signal)
                        
                        # For each wavelength in this animal
                        for channel_num, cols, wavelength in target_columns:
                            if not cols:
                                continue
                            
                            dff_key = f"{channel_num}_{wavelength}"
                            dff_data = animal_data['dff_data'].get(dff_key)
                            if dff_data is None:
                                continue
                            
                            event_time_rel, event_zscore, _ = self.compute_event_zscore_from_dff(
                                dff_data, time_data, start_time, end_time,
                                group_params['pre_window'], group_params['post_window'], post_flag=True
                            )
                            
                            if len(event_time_rel) > 0:
                                # Apply smoothing if requested
                                if group_params['smooth_method'] == "Moving Average":
                                    kernel = np.ones(group_params['smooth_window']) / group_params['smooth_window']
                                    event_zscore = np.convolve(event_zscore, kernel, mode='same')
                                elif group_params['smooth_method'] == "Savitzky-Golay":
                                    event_zscore = savgol_filter(event_zscore, group_params['smooth_window'], 3)
                                
                                # Interpolate to common time axis
                                if len(event_zscore) == len(event_time_rel):
                                    interp_zscore = np.interp(common_time, event_time_rel, event_zscore)
                                    heatmap_data.append(interp_zscore)
                                    # Include wavelength in label
                                    heatmap_y_labels.append(f"{event_label} - {animal_data['animal_id']} CH{channel_num} {wavelength}nm")
                                    y_position += 1
                    
                    # Add event boundary after all animals and channels for this event
                    event_boundaries.append(y_position)
                
                # Plot heatmap
                if heatmap_data:
                    # Ensure all rows have the same length
                    min_length = min([len(row) for row in heatmap_data])
                    heatmap_data = [row[:min_length] for row in heatmap_data]
                    
                    heatmap_array = np.array(heatmap_data)
                    
                    # Create custom colormap
                    colors_map = LinearSegmentedColormap.from_list("custom", ["blue", "white", "red"], N=256)
                    
                    im = ax2.imshow(heatmap_array, aspect='auto', cmap=colors_map,
                                extent=[common_time[0], common_time[-1], 0, len(heatmap_data)],
                                origin='lower')
                    
                    # Add red rectangles for event boundaries
                    # for boundary in event_boundaries[:-1]:
                    #     ax2.axhline(y=boundary, color='red', linewidth=2)
                    
                    ax2.set_xlim(common_time.min(), common_time.max())
                    ax2.set_xlabel("Time (s)", fontsize=12)
                    ax2.set_ylabel("Trials", fontsize=12)
                    # ax2.set_yticks(np.arange(len(heatmap_y_labels)) + 0.5)
                    ax2.tick_params(axis='both', labelsize=10)
                    # ax2.set_yticklabels(heatmap_y_labels, fontsize=8)
                    ax2.set_title(f"Heatmap - Group {group_name} ({event_name}, Target: {target_signal})", fontsize=12, fontweight='bold')
                    
                    # Mark shock positions
                    for animal_data in group_animals:
                        if 'event_data' in animal_data:
                            events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                            events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                            for _, event2 in events2.iterrows():
                                if animal_data.get('event_time_absolute', False):
                                    start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                                    end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    start2 = event2['start_time']
                                    end2 = event2['end_time']
                                
                                for _, event1 in events.iterrows():
                                    if animal_data.get('event_time_absolute', False):
                                        event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                                        event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                                    else:
                                        event1_start = event1['start_time']
                                        event1_end = event1['end_time']
                                    
                                    if start2 >= event1_start and end2 <= event1_end:
                                        rel_start = start2 - event1_start
                                        ax2.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1)
                    
                    # Add colorbar
                    fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.2, label='Z-Score')
                    
                    # Add vertical lines for event start and end
                    ax2.axvline(0, color='#000000', linestyle='--', linewidth=1)
                    ax2.axvline(duration, color='#000000', linestyle='--', linewidth=1)
                
                fig.tight_layout()
                canvas.draw()
            
            self.set_status(f"Event activity and heatmaps plotted for all groups (target: {target_signal})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot: {str(e)}")
            traceback.print_exc()

    def plot_experiment_activity(self, all_group_params, preview=False):
        """Plot experiment-related activity - each wavelength separately"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
        
        try:
            # Clear existing tabs
            experiment_activity_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "experiment activity":
                    experiment_activity_tab_index = i
                    break
            
            if experiment_activity_tab_index is not None:
                self.plot_notebook.forget(experiment_activity_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
            
            experiment_activity_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(experiment_activity_tab, text="experiment activity")
            
            target_signal = self.target_signal_var.get()

            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            group_notebook = ttk.Notebook(experiment_activity_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Define wavelength colors
            wavelength_colors = {
                '410': '#0000FF',
                '470': '#00FF00',
                '560': '#FF0000',
            }
            
            # Process each group that has parameters
            for group_name, group_params in all_group_params.items():
                if group_name not in groups:
                    continue
                    
                group_animals = groups[group_name]
                
                # Create tab for this group
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create figure with 2 subplots
                fig = Figure(figsize=(10, 8), dpi=100)
                canvas = FigureCanvasTkAgg(fig, master=group_tab)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                
                toolbar_frame = ttk.Frame(group_tab)
                toolbar_frame.pack(fill="x")
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                
                # Create subplots
                ax1 = fig.add_subplot(2, 1, 1)  # dF/F
                ax2 = fig.add_subplot(2, 1, 2)  # Z-score
                
                event_type = group_params['event_type']
                event_name = "Sound" if event_type == 1 else "Shock"
                
                # Collect all responses for dF/F and Z-score - SEPARATED BY WAVELENGTH
                all_dff_responses = {}  # key: wavelength, value: list of responses
                all_zscore_responses = {}  # key: wavelength, value: list of responses

                # Get event indices for selected events
                selected_event_indices = {}
                for animal_data in group_animals:
                    if 'event_data' in animal_data:
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                        for event_idx, (_, event) in enumerate(events.iterrows()):
                            event_label = f"{event_name}{event_idx+1}"
                            if event_label in group_params['selected_events']:
                                selected_event_indices[event_label] = event_idx

                # Sort events by label
                sorted_event_labels = sorted(selected_event_indices.keys(), 
                                        key=lambda x: int(x.replace(event_name, "")))

                # Process each selected event
                for event_label_idx in sorted_event_labels:
                    event_idx = selected_event_indices[event_label_idx]
                    
                    # Collect all responses for this event (all animals and channels)
                    for animal_data in group_animals:
                        if 'dff_data' not in animal_data or 'event_data' not in animal_data:
                            continue
                        
                        # Get event data
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                        if event_idx >= len(events):
                            continue
                        
                        event = events.iloc[event_idx]
                        
                        time_col = animal_data['channels']['time']
                        time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                        
                        if animal_data.get('event_time_absolute', False):
                            start_time = event['start_absolute'] - animal_data['video_start_fiber']
                            end_time = event['end_absolute'] - animal_data['video_start_fiber']
                        else:
                            start_time = event['start_time']
                            end_time = event['end_time']
                        
                        duration = end_time - start_time

                        target_columns = self.get_target_columns(animal_data, target_signal)
                        
                        # Process each wavelength separately
                        for channel_num, cols, wavelength in target_columns:
                            if not cols:
                                continue
                            
                            dff_key = f"{channel_num}_{wavelength}"
                            dff_data = animal_data['dff_data'].get(dff_key)
                            if dff_data is None:
                                continue
                            
                            # Compute event dF/F
                            event_time_rel, event_dff = self.compute_event_dff_from_dff(
                                dff_data, time_data, start_time, end_time,
                                group_params['pre_window'], group_params['post_window'], post_flag=True
                            )

                            # Compute event Z-score
                            event_time_rel, event_zscore, _ = self.compute_event_zscore_from_dff(
                                dff_data, time_data, start_time, end_time,
                                group_params['pre_window'], group_params['post_window'], post_flag=True
                            )

                            if len(event_time_rel) > 0:
                                # Apply smoothing
                                if group_params['smooth_method'] == "Moving Average":
                                    kernel = np.ones(group_params['smooth_window']) / group_params['smooth_window']
                                    event_dff_smooth = np.convolve(event_dff, kernel, mode='same')
                                    event_zscore_smooth = np.convolve(event_zscore, kernel, mode='same')
                                elif group_params['smooth_method'] == "Savitzky-Golay":
                                    event_dff_smooth = savgol_filter(event_dff, group_params['smooth_window'], 3)
                                    event_zscore_smooth = savgol_filter(event_zscore, group_params['smooth_window'], 3)
                                else:
                                    event_dff_smooth = event_dff
                                    event_zscore_smooth = event_zscore
                                
                                # Add each wavelength separately
                                if wavelength not in all_dff_responses:
                                    all_dff_responses[wavelength] = []
                                if wavelength not in all_zscore_responses:
                                    all_zscore_responses[wavelength] = []
                                
                                all_dff_responses[wavelength].append((event_time_rel, event_dff_smooth))
                                all_zscore_responses[wavelength].append((event_time_rel, event_zscore_smooth))
                
                if not all_dff_responses:
                    messagebox.showwarning("No Data", f"No valid data to plot for group {group_name}")
                    continue
                
                # Calculate common time using fps_fiber
                all_time_data = []
                for responses in all_dff_responses.values():
                    for t, _ in responses:
                        all_time_data.extend(t)
                
                if all_time_data:
                    max_time = max(all_time_data)
                    min_time = min(all_time_data)
                    common_time = np.linspace(min_time, max_time, int((max_time - min_time) * fps_fiber / group_params['downsample_rate']))
                else:
                    common_time = np.linspace(-group_params['pre_window'], group_params['post_window'], 100)
                
                # Plot dF/F for each wavelength separately
                for wavelength in sorted(all_dff_responses.keys()):
                    responses = all_dff_responses[wavelength]
                    
                    # Interpolate to common time axis
                    common_dff_responses = []
                    for time_rel, dff in responses:
                        if len(time_rel) > 0 and len(dff) == len(time_rel):
                            interp_dff = np.interp(common_time, time_rel, dff)
                            common_dff_responses.append(interp_dff)
                    
                    if common_dff_responses:
                        mean_dff = np.nanmean(common_dff_responses, axis=0)
                        sem_dff = np.nanstd(common_dff_responses, axis=0, ddof=1) / np.sqrt(len(common_dff_responses))
                        
                        color = wavelength_colors.get(wavelength, "#808080")
                        
                        ax1.plot(common_time, mean_dff, color=color, linestyle='-', linewidth=2, 
                                label=f'{wavelength}nm (n={len(common_dff_responses)})')
                        ax1.fill_between(common_time, mean_dff - sem_dff, mean_dff + sem_dff, 
                                        color=color, alpha=0.2)
                
                # Mark event positions for dF/F plot
                for animal_data in group_animals:
                    if 'event_data' in animal_data:
                        events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                        for _, event2 in events2.iterrows():
                            if animal_data.get('event_time_absolute', False):
                                start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                                end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                            else:
                                start2 = event2['start_time']
                                end2 = event2['end_time']
                            
                            for _, event1 in events.iterrows():
                                if animal_data.get('event_time_absolute', False):
                                    event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                                    event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    event1_start = event1['start_time']
                                    event1_end = event1['end_time']
                                
                                if start2 >= event1_start and end2 <= event1_end:
                                    rel_start = start2 - event1_start
                                    ax1.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1,
                                            label='Shock Start' if 'Shock Start' not in ax1.get_legend_handles_labels()[1] else "")
                
                ax1.axvline(0, color='k', linestyle='--', linewidth=1, label=f'{event_name} Start')
                ax1.axvline(duration, color='k', linestyle='--', linewidth=1, label=f'{event_name} End')
                ax1.axhline(0, color='#888888', linestyle='--', linewidth=1)
                ax1.set_xlim(common_time.min(), common_time.max())
                ax1.set_ylabel("ΔF/F")
                ax1.set_title(f"Experiment-Related Activity (ΔF/F) - Group {group_name} ({event_name}, Target: {target_signal})")
                ax1.grid(False)
                # ax1.legend()
                
                # Plot Z-score for each wavelength separately
                for wavelength in sorted(all_zscore_responses.keys()):
                    responses = all_zscore_responses[wavelength]
                    
                    # Interpolate to common time axis
                    common_zscore_responses = []
                    for time_rel, zscore in responses:
                        if len(time_rel) > 0 and len(zscore) == len(time_rel):
                            interp_zscore = np.interp(common_time, time_rel, zscore)
                            common_zscore_responses.append(interp_zscore)
                    
                    if common_zscore_responses:
                        mean_zscore = np.nanmean(common_zscore_responses, axis=0)
                        sem_zscore = np.nanstd(common_zscore_responses, axis=0, ddof=1) / np.sqrt(len(common_zscore_responses))
                        
                        color = wavelength_colors.get(wavelength, "#808080")
                        
                        ax2.plot(common_time, mean_zscore, color=color, linestyle='-', linewidth=2, 
                                label=f'{wavelength}nm (n={len(common_zscore_responses)})')
                        ax2.fill_between(common_time, mean_zscore - sem_zscore, mean_zscore + sem_zscore, 
                                        color=color, alpha=0.2)
                
                # Mark event positions for Z-score plot
                for animal_data in group_animals:
                    if 'event_data' in animal_data:
                        events2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                        events = animal_data['event_data'][animal_data['event_data']['Event Type'] == event_type]
                        for _, event2 in events2.iterrows():
                            if animal_data.get('event_time_absolute', False):
                                start2 = event2['start_absolute'] - animal_data['video_start_fiber']
                                end2 = event2['end_absolute'] - animal_data['video_start_fiber']
                            else:
                                start2 = event2['start_time']
                                end2 = event2['end_time']
                            
                            for _, event1 in events.iterrows():
                                if animal_data.get('event_time_absolute', False):
                                    event1_start = event1['start_absolute'] - animal_data['video_start_fiber']
                                    event1_end = event1['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    event1_start = event1['start_time']
                                    event1_end = event1['end_time']
                                
                                if start2 >= event1_start and end2 <= event1_end:
                                    rel_start = start2 - event1_start
                                    ax2.axvline(rel_start, color="#FF0000", linestyle='--', linewidth=1,
                                            label='Shock Start' if 'Shock Start' not in ax2.get_legend_handles_labels()[1] else "")
                
                ax2.axvline(0, color='k', linestyle='--', linewidth=1, label=f'{event_name} Start')
                ax2.axvline(duration, color='k', linestyle='--', linewidth=1, label=f'{event_name} End')
                ax2.axhline(0, color='#888888', linestyle='--', linewidth=1)
                ax2.set_xlim(common_time.min(), common_time.max())
                ax2.set_ylim([-4, 2])
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Z-Score")
                ax2.set_title(f"Experiment-Related Activity (Z-Score) - Group {group_name} ({event_name}, Target: {target_signal})")
                ax2.grid(False)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                # ax2.legend()
                
                fig.tight_layout()
                canvas.draw()
            
            self.set_status(f"Experiment activity plotted for all groups (target: {target_signal})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot: {str(e)}")
            traceback.print_exc()

    def export_statistic_results(self):
        """Export statistics for selected animals with group-based processing"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Check if any animal has event type 2
        has_type2 = False
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'event_data' in animal_data:
                if 2 in animal_data['event_data']['Event Type'].values:
                    has_type2 = True
                    break
        
        target_signal = self.target_signal_var.get()

        # Group animals by group
        groups = {}
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            group = animal_data.get('group', 'Unknown')
            if group not in groups:
                groups[group] = []
            groups[group].append(animal_data)
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Statistics")
        dialog.geometry("250x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_frame, text="Time Windows (seconds):").grid(row=0, column=0, columnspan=2, sticky="w", pady=10)
        
        if has_type2:
            ttk.Label(main_frame, text="Pre Event Type 1 Start:").grid(row=1, column=0, sticky="w", pady=5)
            pre_start = ttk.Entry(main_frame, width=10)
            pre_start.insert(0, "5")
            pre_start.grid(row=1, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Post Event Type 1 Start:").grid(row=2, column=0, sticky="w", pady=5)
            post_start = ttk.Entry(main_frame, width=10)
            post_start.insert(0, "5")
            post_start.grid(row=2, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Pre Event Type 2 Start:").grid(row=3, column=0, sticky="w", pady=5)
            pre_type2 = ttk.Entry(main_frame, width=10)
            pre_type2.insert(0, "5")
            pre_type2.grid(row=3, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Post Event Type 1 End:").grid(row=4, column=0, sticky="w", pady=5)
            post_end = ttk.Entry(main_frame, width=10)
            post_end.insert(0, "5")
            post_end.grid(row=4, column=1, sticky="w")
        else:
            ttk.Label(main_frame, text="Pre Event Type 1 Start:").grid(row=1, column=0, sticky="w", pady=5)
            pre_start = ttk.Entry(main_frame, width=10)
            pre_start.insert(0, "5")
            pre_start.grid(row=1, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Post Event Type 1 Start:").grid(row=2, column=0, sticky="w", pady=5)
            post_start = ttk.Entry(main_frame, width=10)
            post_start.insert(0, "5")
            post_start.grid(row=2, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Pre Event Type 1 End:").grid(row=3, column=0, sticky="w", pady=5)
            pre_end = ttk.Entry(main_frame, width=10)
            pre_end.insert(0, "5")
            pre_end.grid(row=3, column=1, sticky="w")
            
            ttk.Label(main_frame, text="Post Event Type 1 End:").grid(row=4, column=0, sticky="w", pady=5)
            post_end = ttk.Entry(main_frame, width=10)
            post_end.insert(0, "5")
            post_end.grid(row=4, column=1, sticky="w")
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        def export():
            try:
                all_stats = []
                
                for group_name, group_animals in groups.items():
                    for animal_data in group_animals:
                        if 'dff_data' not in animal_data or 'event_data' not in animal_data:
                            continue
                        
                        animal_id = animal_data['animal_id']
                        group = animal_data.get('group', 'Unknown')
                        
                        events_type1 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 1]
                        if events_type1.empty:
                            continue
                        
                        time_col = animal_data['channels']['time']
                        time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                        
                        target_columns = self.get_target_columns(animal_data, target_signal)

                        for channel_num, cols, wavelength in target_columns:
                            if not cols:
                                continue
                            
                            dff_key = f"{channel_num}_{wavelength}"
                            dff_data = animal_data['dff_data'].get(dff_key)
                            if dff_data is None:
                                continue
                            
                            for event_idx, (_, event) in enumerate(events_type1.iterrows()):
                                if animal_data.get('event_time_absolute', False):
                                    start_time = event['start_absolute'] - animal_data['video_start_fiber']
                                    end_time = event['end_absolute'] - animal_data['video_start_fiber']
                                else:
                                    start_time = event['start_time']
                                    end_time = event['end_time']
                                
                                # Calculate z-score for this specific event using pre-event window as baseline
                                pre_window_start = start_time - float(pre_start.get())
                                pre_window_end = start_time
                                
                                pre_window_mask = (time_data >= pre_window_start) & (time_data <= pre_window_end)
                                pre_window_dff = dff_data[pre_window_mask]
                                # print(pre_window_mask)
                                
                                if len(pre_window_dff) < 2:
                                    baseline_mean = np.nan
                                    baseline_std = np.nan
                                else:
                                    baseline_mean = np.mean(pre_window_dff)
                                    baseline_std = np.std(pre_window_dff)
                                
                                if has_type2:
                                    # 4 windows + type 2 period
                                    windows = [
                                        ('Pre Type1 Start', start_time - float(pre_start.get()), start_time),
                                        ('Post Event Type1 Start', start_time, start_time + float(post_start.get())),
                                        ('Pre Type2 Start', None, None),  # Will be filled if type2 exists
                                        ('Post Event Type1 End', end_time, end_time + float(post_end.get()))
                                    ]
                                    
                                    # Find type 2 events during this type 1 event
                                    events_type2 = animal_data['event_data'][animal_data['event_data']['Event Type'] == 2]
                                    for _, event2 in events_type2.iterrows():
                                        if animal_data.get('event_time_absolute', False):
                                            type2_start = event2['start_absolute'] - animal_data['video_start_fiber']
                                            type2_end = event2['end_absolute'] - animal_data['video_start_fiber']
                                        else:
                                            type2_start = event2['start_time']
                                            type2_end = event2['end_time']
                                        
                                        if type2_start >= start_time and type2_end <= end_time:
                                            windows[2] = ('Pre Type2 Start', type2_start - float(pre_type2.get()), type2_start)
                                            windows.append(('Type2 Period', type2_start, type2_end))
                                            break
                                else:
                                    # 4 windows
                                    windows = [
                                        ('Pre Type1 Start', start_time - float(pre_start.get()), start_time),
                                        ('Post Event Type1 Start', start_time, start_time + float(post_start.get())),
                                        ('Pre Event Type1 End', end_time - float(pre_end.get()), end_time),
                                        ('Post Event Type1 End', end_time, end_time + float(post_end.get()))
                                    ]
                                
                                for window_name, win_start, win_end in windows:
                                    if win_start is None or win_end is None:
                                        continue
                                    
                                    mask = (time_data >= win_start) & (time_data <= win_end)
                                    window_dff = dff_data[mask]
                                    window_time = time_data[mask]

                                    if len(window_dff) > 0:
                                        mean_val_dff = float(np.mean(window_dff))
                                        max_val_dff = float(np.max(window_dff))
                                        min_val_dff = float(np.min(window_dff))
                                        try:
                                            auc_val_dff = float(np.trapezoid(window_dff, window_time))
                                        except AttributeError:
                                            auc_val_dff = float(np.trapz(window_dff, window_time))
                                        
                                    # Calculate z-score for this window using the pre-event baseline
                                    if not np.isnan(baseline_mean) and not np.isnan(baseline_std) and baseline_std > 0:
                                        window_zscore = (window_dff - baseline_mean) / baseline_std
                                        mean_val_zscore = float(np.mean(window_zscore))
                                        max_val_zscore = float(np.max(window_zscore))
                                        min_val_zscore = float(np.min(window_zscore))
                                        try:
                                            auc_val_zscore = float(np.trapezoid(window_zscore, window_time))
                                        except AttributeError:
                                            auc_val_zscore = float(np.trapz(window_zscore, window_time))
                                    else:
                                        mean_val_zscore = np.nan
                                        max_val_zscore = np.nan
                                        min_val_zscore = np.nan
                                        auc_val_zscore = np.nan
                                    
                                    all_stats.append({
                                        'Group': group,
                                        'Animal ID': animal_id,
                                        'Channel': channel_num,
                                        'Wavelength': wavelength,
                                        'Target Signal': target_signal,
                                        'Event Index': event_idx + 1,
                                        'Window': window_name,
                                        'Mean(dff)': mean_val_dff,
                                        'Max(dff)': max_val_dff,
                                        'Min(dff)': min_val_dff,
                                        'AUC(dff)': auc_val_dff,
                                        'Mean(z-score)': mean_val_zscore,
                                        'Max(z-score)': max_val_zscore,
                                        'Min(z-score)': min_val_zscore,
                                        'AUC(z-score)': auc_val_zscore
                                    })
                
                if not all_stats:
                    messagebox.showinfo("No Data", "No valid statistics calculated")
                    return
                
                df = pd.DataFrame(all_stats)
                
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    title="Save Statistics Results"
                )
                
                if save_path:
                    df.to_csv(save_path, index=False, float_format='%.10f')
                    messagebox.showinfo("Success", f"Statistics exported to:\n{save_path}")
                    self.set_status(f"Statistics exported: {len(all_stats)} records (target: {target_signal})")
                    dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")
                traceback.print_exc()
        
        ttk.Button(btn_frame, text="Export", command=export).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

class FreezingAnalyzerApp(BaseAnalyzerApp):
    def __init__(self, root, include_fiber=False):
        super().__init__(root, include_fiber)
        self.root.title("ML Lab Freezing Analyzer")
        self.root.state('zoomed')
        self.dlc_results_path = ""
        self.events_path = ""
        self.freezing_data = None
        self.window_freeze = None
        self.fps = 30
        self.raw_speed = None
        self.interp_speed = None
        self.smooth_speed = None
        self.freeze_threshold = None
        self.freeze_idx = None
        self.dlc_freeze_status = None
        self.multi_animal_results = {}
        self.cam_id = None
        self.create_ui()

    def create_main_controls(self):
        control_panel = ttk.Frame(self.control_frame)
        control_panel.pack(fill="x", padx=5, pady=5)

        # Multi-animal list frame
        multi_animal_frame = ttk.LabelFrame(control_panel, text="Multi Animal Analysis")
        multi_animal_frame.pack(fill="x", padx=5, pady=5)
        
        list_frame = ttk.Frame(multi_animal_frame)
        list_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, height=5)
        self.file_listbox.pack(fill=tk.X, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        btn_frame = ttk.Frame(multi_animal_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Single Import", command=self.single_import).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Clear Selected", command=self.clear_selected).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Multi Import", command=self.multi_import).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Select All", command=self.select_all).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        # Freezing analysis buttons
        freezing_frame = ttk.LabelFrame(control_panel, text="Freezing Analysis")
        freezing_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(freezing_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        ttk.Button(button_frame, text="Plot Raster", command=self.plot_raster_with_event_selection).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Plot Freezing Timeline", command=self.plot_freezing_timeline).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Export Freezing", command=self.export_freezing_csv).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Compare Raw vs Cleaned", command=self.plot_compare_raw_cleaning).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        if self.include_fiber:
            fiber_frame = ttk.LabelFrame(control_panel, text="Fiber Photometry")
            fiber_frame.pack(fill="x", padx=5, pady=5)
            fiber_frame.columnconfigure(0, weight=1)
            fiber_frame.columnconfigure(1, weight=1)
            
            ttk.Button(fiber_frame, text="Plot Raw Data", command=self.plot_raw_data).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
            ttk.Button(fiber_frame, text="Plot Freezing w/Fiber", command=self.plot_freezing_with_fiber).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
    
    def show_step1(self):
        self.set_status("Step 1: Data Loading")
    
    def show_step2(self):
        self.set_status("Step 2: Preprocessing")

        if not hasattr(self, 'step2_frame') or not self.step2_frame:
            self.step2_frame = ttk.LabelFrame(self.control_frame, text="Preprocessing")
            self.step2_frame.pack(fill="x", padx=5, pady=5)
        
        self.step2_frame.pack(fill="x", padx=5, pady=5)
        self.step2_frame.update_idletasks()
        
        for widget in self.step2_frame.winfo_children():
            widget.destroy()
        
        signal_frame = ttk.Frame(self.step2_frame)
        signal_frame.pack(fill="x", padx=5, pady=5)

        available_targets = self.get_available_target_signals()
        if not available_targets:
            available_targets = ["470", "560"]
        
        ttk.Label(signal_frame, text="Target Signal:").grid(row=0, column=0, sticky="w")
        target_menu = ttk.OptionMenu(signal_frame, self.target_signal_var, available_targets[0], *available_targets)
        target_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        ttk.Label(signal_frame, text="Reference Signal:").grid(row=1, column=0, sticky="w")
        ref_options = ["410", "baseline"]
        ref_menu = ttk.OptionMenu(signal_frame, self.reference_signal_var, "410", *ref_options)
        ref_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        self.baseline_frame = ttk.Frame(self.step2_frame)
        self.baseline_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(self.baseline_frame, text="Baseline Period (s):").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.baseline_frame, textvariable=self.baseline_start, width=6).grid(row=0, column=1, padx=2)
        ttk.Label(self.baseline_frame, text="to").grid(row=0, column=2)
        ttk.Entry(self.baseline_frame, textvariable=self.baseline_end, width=6).grid(row=0, column=3, padx=2)
        
        if self.reference_signal_var.get() != "baseline":
            self.baseline_frame.pack_forget()
        
        smooth_frame = ttk.Frame(self.step2_frame)
        smooth_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(smooth_frame, text="Apply Smoothing", variable=self.apply_smooth,
                       command=lambda: self.toggle_widgets(smooth_frame, self.apply_smooth.get(), 1)).grid(row=0, column=0, sticky="w")
        
        param_frame = ttk.Frame(smooth_frame)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Label(param_frame, text="Window Size:").grid(row=0, column=0, sticky="w")
        ttk.Scale(param_frame, from_=3, to=101, orient=tk.HORIZONTAL, 
                 length=100, variable=self.smooth_window,
                 command=lambda v: self.smooth_window.set(int(float(v)))).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_window).grid(row=0, column=2)
        
        ttk.Label(param_frame, text="Polynomial Order:").grid(row=1, column=0, sticky="w")
        ttk.Scale(param_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                 length=100, variable=self.smooth_order,
                 command=lambda v: self.smooth_order.set(int(float(v)))).grid(row=1, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_order).grid(row=1, column=2)
        
        param_frame.grid_remove()
        
        baseline_corr_frame = ttk.Frame(self.step2_frame)
        baseline_corr_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(baseline_corr_frame, text="Apply Baseline Correction", variable=self.apply_baseline,
                        command=lambda: self.toggle_widgets(baseline_corr_frame, self.apply_baseline.get(), 1)).grid(row=0, column=0, sticky="w")
        
        model_frame = ttk.Frame(baseline_corr_frame)
        model_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        ttk.Label(model_frame, text="Baseline Model:").grid(row=1, column=0, sticky="w")
        model_options = ["Polynomial", "Exponential"]
        model_menu = ttk.OptionMenu(model_frame, self.baseline_model, "Polynomial", *model_options)
        model_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        model_frame.grid_remove()
        
        motion_frame = ttk.Frame(self.step2_frame)
        motion_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(motion_frame, text="Apply Motion Correction", variable=self.apply_motion,
                       command=lambda: self.toggle_motion_correction()).grid(row=0, column=0, sticky="w")
        
        ttk.Button(self.step2_frame, text="Apply Preprocessing", 
                  command=self.apply_preprocessing).pack(pady=10)
        
        self.reference_signal_var.trace_add("write", self.update_baseline_ui)

    def get_available_target_signals(self):
        all_combinations = set()
        
        for animal_data in self.multi_animal_data:
            if 'wavelength_combinations' in animal_data:
                for combo in animal_data['wavelength_combinations']:
                    all_combinations.add(combo)
            elif 'available_wavelengths' in animal_data:
                for wl in animal_data['available_wavelengths']:
                    all_combinations.add(wl)
        
        return sorted(all_combinations, key=lambda x: (len(x), x))

    def update_baseline_ui(self, *args):
        if self.reference_signal_var.get() == "baseline":
            self.baseline_frame.pack(fill="x", padx=5, pady=5)
        else:
            self.baseline_frame.pack_forget()
    
    def toggle_motion_correction(self):
        if self.reference_signal_var.get() != "410":
            self.apply_motion.set(False)
            messagebox.showinfo("Info", "Motion correction requires 410nm as reference signal")
    
    def toggle_widgets(self, parent_frame, show, index):
        children = parent_frame.winfo_children()
        if len(children) > index:
            if show:
                children[index].grid()
            else:
                children[index].grid_remove()
    
    def show_step3(self):
        self.set_status("Step 3: ΔF/F & Z-score")
        
        if not hasattr(self, 'step3_frame') or not self.step3_frame:
            self.step3_frame = ttk.LabelFrame(self.control_frame, text="ΔF/F & Z-score")
            self.step3_frame.pack(fill="x", padx=5, pady=5)
        
        self.step3_frame.pack(fill="x", padx=5, pady=5)
        self.step3_frame.update_idletasks()
        
        for widget in self.step3_frame.winfo_children():
            widget.destroy()

        self.step3_frame.columnconfigure(0, weight=1)
        self.step3_frame.columnconfigure(1, weight=1)
        
        ttk.Button(self.step3_frame, text="Calculate ΔF/F", command=self.calculate_and_plot_dff).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(self.step3_frame, text="Calculate Z-score", command=self.calculate_and_plot_zscore).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
    
    def show_step4(self):
        self.set_status("Step 4: Event Analysis")

        if not hasattr(self, 'step4_frame') or not self.step4_frame:
            self.step4_frame = ttk.LabelFrame(self.control_frame, text="Event Analysis")
            self.step4_frame.pack(fill="x", padx=5, pady=5)
        
        self.step4_frame.pack(fill="x", padx=5, pady=5)
        self.step4_frame.update_idletasks()
        
        for widget in self.step4_frame.winfo_children():
            widget.destroy()
        
        btn_frame = ttk.Frame(self.step4_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame, text="Plot Event-Activity & Heatmap", 
                  command=self.show_event_activity_dialog).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Plot Experiment-Activity", 
                  command=self.show_experiment_activity_dialog).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Export Statistics", 
                  command=self.export_statistic_results).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

    def calculate_fps_from_fiber(self):
        """Calculate FPS from DLC frames and fiber video duration"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'dlc_data' in animal_data and 'video_start_fiber' in animal_data and 'video_end_fiber' in animal_data:
                total_frames = len(animal_data['dlc_data'])
                video_duration = animal_data['video_end_fiber'] - animal_data['video_start_fiber']
                if video_duration > 0:
                    animal_data['fps'] = total_frames / video_duration
                    print(f"FPS for {animal_data['animal_id']}: {animal_data['fps']:.2f}")
                    self.set_status(f"FPS for {animal_data['animal_id']}: {animal_data['fps']:.2f}")

    def compute_freezing_for_animal(self, animal_data):
        """Compute freezing analysis for a single animal"""
        try:
            dlc_result = animal_data['dlc_data']
            event_data = animal_data['event_data']
            fps = animal_data.get('fps', 30)
            
            scorer = dlc_result.columns.levels[0][0]
            left_ear = 'left_ear'
            right_ear = 'right_ear'
            tail_base = 'tail_base'

            le_x = dlc_result.loc[:, (scorer, left_ear, 'x')].values.astype(float)
            le_y = dlc_result.loc[:, (scorer, left_ear, 'y')].values.astype(float)
            le_r = dlc_result.loc[:, (scorer, left_ear, 'likelihood')].values.astype(float)
            
            re_x = dlc_result.loc[:, (scorer, right_ear, 'x')].values.astype(float)
            re_y = dlc_result.loc[:, (scorer, right_ear, 'y')].values.astype(float)
            re_r = dlc_result.loc[:, (scorer, right_ear, 'likelihood')].values.astype(float)
            
            tb_x = dlc_result.loc[:, (scorer, tail_base, 'x')].values.astype(float)
            tb_y = dlc_result.loc[:, (scorer, tail_base, 'y')].values.astype(float)
            tb_r = dlc_result.loc[:, (scorer, tail_base, 'likelihood')].values.astype(float)

            def interpolate_low_reliability(values, reliability, threshold=0.99):
                valid_mask = reliability >= threshold
                indices = np.arange(len(values))
                
                if np.sum(valid_mask) > 1:
                    interp_func = interp1d(
                        indices[valid_mask], 
                        values[valid_mask], 
                        kind='linear', 
                        fill_value="extrapolate"
                    )
                    interpolated = interp_func(indices)
                    return interpolated
                else:
                    return values

            le_x_i = interpolate_low_reliability(le_x, le_r, 0.99)
            le_y_i = interpolate_low_reliability(le_y, le_r, 0.99)
            re_x_i = interpolate_low_reliability(re_x, re_r, 0.99)
            re_y_i = interpolate_low_reliability(re_y, re_r, 0.99)
            tb_x_i = interpolate_low_reliability(tb_x, tb_r, 0.99)
            tb_y_i = interpolate_low_reliability(tb_y, tb_r, 0.99)

            dist_le = np.insert(np.sqrt(np.diff(le_x)**2 + np.diff(le_y)**2), 0, 0)
            dist_re = np.insert(np.sqrt(np.diff(re_x)**2 + np.diff(re_y)**2), 0, 0)
            dist_tb = np.insert(np.sqrt(np.diff(tb_x)**2 + np.diff(tb_y)**2), 0, 0)

            dist_le_i = np.insert(np.sqrt(np.diff(le_x_i)**2 + np.diff(le_y_i)**2), 0, 0)
            dist_re_i = np.insert(np.sqrt(np.diff(re_x_i)**2 + np.diff(re_y_i)**2), 0, 0)
            dist_tb_i = np.insert(np.sqrt(np.diff(tb_x_i)**2 + np.diff(tb_y_i)**2), 0, 0)

            centroid_x = (le_x_i + re_x_i + tb_x_i) / 3
            centroid_y = (le_y_i + re_y_i + tb_y_i) / 3

            dx = np.diff(centroid_x)
            dy = np.diff(centroid_y)
            dist = np.sqrt(dx**2 + dy**2)
            dist = np.insert(dist, 0, 0)
            raw_dist = dist

            dist[dist > 200] = np.nan
            valid = ~np.isnan(dist)
            dist_interp = np.interp(np.arange(len(dist)), np.flatnonzero(valid), dist[valid])

            kernel = np.ones(15) / 15
            speed_smooth = np.convolve(dist_interp, kernel, mode='same')

            threshold = 1.20
            freeze_flag = speed_smooth < threshold

            groups = groupby(enumerate(freeze_flag), key=lambda x: x[1])
            freeze_idx = []
            for k, g in groups:
                g = list(g)
                if k and len(g) >= 15:
                    idxs = [i for i, _ in g]
                    freeze_idx.extend(idxs)

            # Calculate freezing by event periods
            freezing_results = []
            for _, event in event_data.iterrows():
                event_type = event['Event Type']
                if event_type not in [0, 1, 3, 4]:  # Only process wait, sound, wait-optogenetics, sound-optogenetics
                    continue
                
                start_frame = int(event['start_time'] * fps)
                end_frame = int(event['end_time'] * fps)
                
                event_frames = np.arange(start_frame, min(end_frame, len(speed_smooth)))
                freeze_frames = np.intersect1d(event_frames, freeze_idx)
                
                total_time = len(event_frames) / fps
                freeze_time = len(freeze_frames) / fps
                freeze_percent = (len(freeze_frames) / len(event_frames) * 100) if len(event_frames) > 0 else 0
                
                freezing_results.append({
                    'event_type': event_type,
                    'start_time': event['start_time'],
                    'end_time': event['end_time'],
                    'total_time': total_time,
                    'freeze_time': freeze_time,
                    'freeze_percent': freeze_percent
                })

            dlc_freeze_status = np.zeros(len(speed_smooth), dtype=int)
            dlc_freeze_status[freeze_idx] = 1

            return {
                'freezing_results': freezing_results,
                'dist_le': dist_le,
                'dist_re': dist_re,
                'dist_tb': dist_tb,
                'dist_le_i': dist_le_i,
                'dist_re_i': dist_re_i,
                'dist_tb_i': dist_tb_i,
                'raw_speed': raw_dist,
                'interp_speed': dist_interp,
                'smooth_speed': speed_smooth,
                'freeze_threshold': threshold,
                'freeze_idx': freeze_idx,
                'dlc_freeze_status': dlc_freeze_status
            }
            
        except Exception as e:
            self.set_status(f"Error computing freezing: {str(e)}")
            traceback.print_exc()
            return None

    def plot_raster_with_event_selection(self):
        """Plot raster with event selection - updated to group-based event selection"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Calculate FPS and compute freezing
        self.calculate_fps_from_fiber()
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)
        
        # Group animals by group and get available events for each group
        groups = {}
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            group = animal_data.get('group', 'Unknown')
            if group not in groups:
                groups[group] = []
            groups[group].append(animal_data)
        
        # Get available events for each group (excluding event_type == 2)
        group_events = {}
        group_event_indices = {}
        for group_name, group_animals in groups.items():
            if group_animals and 'event_data' in group_animals[0]:
                event_data = group_animals[0]['event_data']
                # Filter out event_type == 2 and event_type == 5
                filtered_events = []
                original_indices = []
                for idx, row in event_data.iterrows():
                    event_type = row['Event Type']
                    if event_type in [0, 1, 3, 4]:  # Only include wait and sound events, exclude shock (type 2)
                        if event_type == 0:
                            label = "wait"
                        elif event_type == 1:
                            label = "sound"
                        elif event_type == 3:
                            label = "wait-optogenetics"
                        elif event_type == 4:
                            label = "sound-optogenetics"
                        filtered_events.append(f"{label}{math.ceil((idx+1)/2)}")
                        original_indices.append(idx)
                group_events[group_name] = filtered_events
                group_event_indices[group_name] = original_indices
        
        if not group_events:
            messagebox.showwarning("No Events", "No valid events found (only type 0 and 1 are supported)")
            return
        
        # Create dialog for event selection with group tabs
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Event Range by Group")
        dialog.geometry("250x270")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Create notebook for group tabs
        group_notebook = ttk.Notebook(main_frame)
        group_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        group_vars = {}  # Store variables for each group
        
        for group_name, events in group_events.items():
            if not events:  # Skip groups with no events
                continue
                
            # Create tab for this group
            group_tab = ttk.Frame(group_notebook)
            group_notebook.add(group_tab, text=group_name)
            
            ttk.Label(group_tab, text=f"Event Selection for {group_name}", 
                     font=("Arial", 10, "bold")).pack(pady=10)
            
            # Event selection frame
            event_frame = ttk.Frame(group_tab)
            event_frame.pack(fill="x", padx=10, pady=10)
            
            ttk.Label(event_frame, text="Start Event:").grid(row=0, column=0, sticky="w", pady=5)
            start_var = tk.StringVar(value=events[0])
            start_menu = ttk.OptionMenu(event_frame, start_var, events[0], *events)
            start_menu.grid(row=0, column=1, sticky="ew", pady=5)
            
            ttk.Label(event_frame, text="End Event:").grid(row=1, column=0, sticky="w", pady=5)
            end_var = tk.StringVar(value=events[-1])
            end_menu = ttk.OptionMenu(event_frame, end_var, events[-1], *events)
            end_menu.grid(row=1, column=1, sticky="ew", pady=5)
            
            # Store variables for this group
            group_vars[group_name] = {
                'start_var': start_var,
                'end_var': end_var,
                'events': events,
                'original_indices': group_event_indices[group_name]
            }
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=20)
        
        def plot():
            group_event_ranges = {}
            for group_name, vars_dict in group_vars.items():
                start_idx_in_filtered = vars_dict['events'].index(vars_dict['start_var'].get())
                end_idx_in_filtered = vars_dict['events'].index(vars_dict['end_var'].get())
                
                start_original_idx = vars_dict['original_indices'][start_idx_in_filtered]
                end_original_idx = vars_dict['original_indices'][end_idx_in_filtered]
                
                group_event_ranges[group_name] = (start_original_idx, end_original_idx)
            
            dialog.destroy()
            self.plot_raster(group_event_ranges)
        
        ttk.Button(btn_frame, text="Plot", command=plot).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

    def plot_raster(self, group_event_ranges):
        """Plot freezing raster for selected animals with group-based visualization and event range filtering"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
        
        try:
            raster_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "raster":
                    raster_tab_index = i
                    break
            
            if raster_tab_index is not None:
                self.plot_notebook.forget(raster_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
            
            raster_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(raster_tab, text="raster")
            
            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            group_notebook = ttk.Notebook(raster_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)

            # Create subplots for each group
            for group_name, group_animals in groups.items():
                # Skip if no event range defined for this group
                if group_name not in group_event_ranges:
                    continue
                    
                start_original_idx, end_original_idx = group_event_ranges[group_name]
                
                # Create tab for this group
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create figure and canvas for this group
                fig = Figure(figsize=(10, 6), dpi=100)
                canvas = FigureCanvasTkAgg(fig, master=group_tab)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                
                toolbar_frame = ttk.Frame(group_tab)
                toolbar_frame.pack(fill="x")
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                
                ax = fig.add_subplot(111)

                y_pos = 0
                y_ticks = []
                y_labels = []

                # Get time range for this group based on selected events
                time_range_start = None
                time_range_end = None
                
                for animal_data in group_animals:
                    if 'event_data' in animal_data:
                        event_data = animal_data['event_data']
                        
                        # Check if the selected indices are within the events range
                        if start_original_idx < len(event_data) and end_original_idx < len(event_data):
                            start_event = event_data.iloc[start_original_idx]
                            end_event = event_data.iloc[end_original_idx]
                            
                            if time_range_start is None or start_event['start_time'] < time_range_start:
                                time_range_start = start_event['start_time']
                            if time_range_end is None or end_event['end_time'] > time_range_end:
                                time_range_end = end_event['end_time']
                
                for animal_data in group_animals:
                    if 'dlc_freeze_status' not in animal_data:
                        continue
                    
                    animal_id = animal_data['animal_id']
                    fps = animal_data.get('fps', 30)
                    
                    y_ticks.append(y_pos)
                    y_labels.append(animal_id)

                    dlc_freeze_status = animal_data['dlc_freeze_status']
                    time_values = np.arange(len(dlc_freeze_status)) / fps
                    
                    # Filter freezing times to only include within selected time range
                    if time_range_start is not None and time_range_end is not None:
                        time_mask = (time_values >= time_range_start) & (time_values <= time_range_end)
                        filtered_time_values = time_values[time_mask]
                        filtered_freeze_status = dlc_freeze_status[time_mask]
                    else:
                        filtered_time_values = time_values
                        filtered_freeze_status = dlc_freeze_status
                    
                    if 'event_data' in animal_data:
                        event_data = animal_data['event_data']
                        
                        # Check if the selected indices are within the events range
                        if start_original_idx < len(event_data) and end_original_idx < len(event_data):
                            for ev_idx in range(start_original_idx, end_original_idx + 1):
                                event = event_data.iloc[ev_idx]
                                event_type = event['Event Type']
                                
                                if event_type == 0:
                                    color = "#ffffff"
                                elif event_type == 1:
                                    color = "#0400fc"
                                elif event_type == 2:
                                    color = "#ff0000"
                                elif event_type == 3:
                                    color = "#00ff00"
                                elif event_type == 4:
                                    color = "#ff00ff"
                                elif event_type == 5:
                                    color = "#ffff00"
                                
                                ax.add_patch(plt.Rectangle(
                                    (event['start_time'], y_pos - 0.4),
                                    event['end_time'] - event['start_time'],
                                    0.8, color=color, alpha=0.5
                                ))
                    
                    # Plot freezing (only within selected time range)
                    freezing_times = filtered_time_values[filtered_freeze_status == 1]
                    for t in freezing_times:
                        ax.vlines(t, y_pos - 0.4, y_pos + 0.4, 
                                color='black', alpha=0.5, linewidth=0.5)
                    
                    y_pos += 1
                
                if y_ticks:
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Animals")
                    
                    # Get event labels for title
                    if group_animals and 'event_data' in group_animals[0]:
                        event_data = group_animals[0]['event_data']
                        if start_original_idx < len(event_data) and end_original_idx < len(event_data):
                            start_event_type = event_data.iloc[start_original_idx]['Event Type']
                            end_event_type = event_data.iloc[end_original_idx]['Event Type']
                            
                            if start_event_type == 0:
                                start_event_label = "wait"
                            elif start_event_type == 1:
                                start_event_label = "sound"
                            elif start_event_type == 2:
                                start_event_label = "shock"
                            elif start_event_type == 3:
                                start_event_label = "wait-optogenetics"
                            elif start_event_type == 4:
                                start_event_label = "sound-optogenetics"
                            elif start_event_type == 5:
                                start_event_label = "shock-optogenetics"
                            
                            if end_event_type == 0:
                                end_event_label = "wait"
                            elif end_event_type == 1:
                                end_event_label = "sound"
                            elif end_event_type == 2:
                                end_event_label = "shock"
                            elif end_event_type == 3:
                                end_event_label = "wait-optogenetics"
                            elif end_event_type == 4:
                                end_event_label = "sound-optogenetics"
                            elif end_event_type == 5:
                                end_event_label = "shock-optogenetics"

                            ax.set_title(f"Freezing Raster Plot - Group {group_name}\n(Events {start_event_label}{math.ceil((start_original_idx+1)/3)} to {end_event_label}{math.ceil((end_original_idx+1)/3)})")
                        else:
                            ax.set_title(f"Freezing Raster Plot - Group {group_name}")
                    else:
                        ax.set_title(f"Freezing Raster Plot - Group {group_name}")
                    
                    ax.grid(False)
                    
                    # Set x-axis limits to show only the selected time range
                    if time_range_start is not None and time_range_end is not None:
                        ax.set_xlim(time_range_start - 5, time_range_end + 5)  # Add some padding
                
                fig.tight_layout()
                canvas.draw()
            
            self.set_status("Freezing raster plotted with event range filtering")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raster: {str(e)}")
            traceback.print_exc()

    def plot_freezing_timeline(self):
        """Plot freezing timeline for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return

        self.calculate_fps_from_fiber()

        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)

        try:
            freezing_timeline_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                if self.plot_notebook.tab(tab, "text") == "freezing timeline":
                    freezing_timeline_tab_index = i
                    break
            if freezing_timeline_tab_index is not None:
                self.plot_notebook.forget(freezing_timeline_tab_index)

            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                if self.plot_notebook.tab(tab, "text") == "Plots":
                    default_tab_index = i
                    break
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)

            freezing_timeline_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(freezing_timeline_tab, text="freezing timeline")

            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                groups.setdefault(group, []).append(animal_data)

            group_notebook = ttk.Notebook(freezing_timeline_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)

            for group_name, group_animals in groups.items():
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)

                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)

                group_data = {}   # key: event_label, value: list of freeze_percent
                for animal_data in group_animals:
                    if 'freezing_results' not in animal_data:
                        continue
                    for i, res in enumerate(animal_data['freezing_results']):
                        event_type = res['event_type']
                        if event_type == 0:
                            label = f"wait{math.ceil((i+1)/2)}"
                        elif event_type == 1:
                            label = f"sound{math.ceil((i+1)/2)}"
                        elif event_type == 3:
                            label = f"wait-optogenetics{math.ceil((i+1)/2)}"
                        elif event_type == 4:
                            label = f"sound-optogenetics{math.ceil((i+1)/2)}"

                        group_data.setdefault(label, []).append(res['freeze_percent'])

                labels_order = list(group_data.keys())
                mean_vals = np.array([np.mean(group_data[l]) for l in labels_order])
                sem_vals  = np.array([np.std(group_data[l])/np.sqrt(len(group_data[l]))
                                    if len(group_data[l]) > 1 else 0 for l in labels_order])

                total_tab = ttk.Frame(animal_notebook)
                animal_notebook.add(total_tab, text="Total")

                fig_total, ax_total = plt.subplots(figsize=(5, 3), dpi=100)
                canvas_total = FigureCanvasTkAgg(fig_total, master=total_tab)
                canvas_total.get_tk_widget().pack(fill="both", expand=True)
                toolbar_frame = ttk.Frame(total_tab); toolbar_frame.pack(fill="x")
                NavigationToolbar2Tk(canvas_total, toolbar_frame).update()

                ax_total.bar(np.arange(len(labels_order)), mean_vals, yerr=sem_vals,
                            capsize=5, edgecolor='white', linewidth=1.2,
                            joinstyle='round', capstyle='round', alpha=0.85, color='#3B75AF')
                ax_total.tick_params(axis='both', which='both', length=0)
                ax_total.set_xticks(np.arange(len(labels_order)))
                ax_total.set_xticklabels(labels_order, fontsize=11, fontweight='normal', rotation=45, ha='right')
                ax_total.set_yticks([0, 25, 50, 75, 100])
                ax_total.set_yticklabels(['0','25','50','75','100'], fontsize=11, fontweight='normal')
                ax_total.set_ylabel("Freezing %", fontsize=14, fontweight='bold')
                ax_total.set_xlabel("Events", fontsize=14, fontweight='bold')
                ax_total.set_title(f" All animal {group_name}  Mean ± SEM of Freezing Timeline by Event", fontsize=16, fontweight='bold')
                ax_total.grid(False)
                fig_total.tight_layout()
                canvas_total.draw()

                for animal_data in group_animals:
                    if 'smooth_speed' not in animal_data:
                        continue
                    animal_id = animal_data['animal_id']
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)

                    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    toolbar_frame = ttk.Frame(animal_tab); toolbar_frame.pack(fill="x")
                    NavigationToolbar2Tk(canvas, toolbar_frame).update()

                    all_labels = []
                    all_percenatges = []
                    for i, result in enumerate(animal_data['freezing_results']):
                        event_type = result['event_type']
                        if event_type == 0:
                            label = f"wait{math.ceil((i+1)/2)}"
                        elif event_type == 1:
                            label = f"sound{math.ceil((i+1)/2)}"
                        elif event_type == 3:
                            label = f"wait-optogenetics{math.ceil((i+1)/2)}"
                        elif event_type == 4:
                            label = f"sound-optogenetics{math.ceil((i+1)/2)}"

                        if label not in all_labels:
                            all_labels.append(label)
                        all_percenatges.append(result['freeze_percent'])

                    ax.bar(np.arange(len(all_labels)), all_percenatges, capsize=5,
                        edgecolor='white', linewidth=1.2, joinstyle='round',
                        capstyle='round', alpha=0.85, color='#3B75AF')
                    ax.tick_params(axis='both', which='both', length=0)
                    ax.set_xticks(np.arange(len(all_labels)))
                    ax.set_xticklabels(all_labels, fontsize=11, fontweight='normal', rotation=45, ha='right')
                    ax.set_xlabel("Events", fontsize=14, fontweight='bold')
                    ax.set_yticks([0, 25, 50, 75, 100])
                    ax.set_yticklabels(['0','25','50','75','100'], fontsize=11, fontweight='normal')
                    ax.set_ylabel("Freezing %", fontsize=14, fontweight='bold')
                    ax.set_title("Freezing Timeline by Event", fontsize=18, fontweight='bold')
                    ax.grid(False)
                    fig.tight_layout()
                    canvas.draw()

            self.set_status("Freezing timeline plotted")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot freezing timeline: {str(e)}")
            traceback.print_exc()

    def plot_freezing_with_fiber(self):
        """Plot freezing with fiber data for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Calculate FPS and compute freezing
        self.calculate_fps_from_fiber()
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)

        try:
            freezing_with_fiber_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "freezing with fiber":
                    freezing_with_fiber_tab_index = i
                    break
            
            if freezing_with_fiber_tab_index is not None:
                self.plot_notebook.forget(freezing_with_fiber_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
            
            freezing_with_fiber_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(freezing_with_fiber_tab, text="freezing with fiber")
            
            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            group_notebook = ttk.Notebook(freezing_with_fiber_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)

            # Create subplots for each group
            for group_name, group_animals in groups.items():
                # Create tab for this group
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create notebook for animals within this group
                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)

                # Plot freezing data
                for animal_idx, animal_data in enumerate(group_animals):
                    if 'smooth_speed' not in animal_data and 'fiber_cropped' not in animal_data:
                        continue
                    
                    animal_id = animal_data['animal_id']
                    
                    # Create tab for this animal
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)

                    # Create figure and canvas for this group
                    fig = Figure(figsize=(10, 8), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(animal_tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    # Create subplots
                    ax1 = fig.add_subplot(2, 1, 1)  # Freezing
                    ax2 = fig.add_subplot(2, 1, 2)  # Fiber data
                    
                    if 'dlc_freeze_status' not in animal_data:
                        continue
                    
                    fps = animal_data.get('fps', 30)
                    dlc_freeze_status = animal_data['dlc_freeze_status']
                    time_values = np.arange(len(dlc_freeze_status)) / fps
                    freezing_times = time_values[dlc_freeze_status == 1]
                    
                    y_pos = animal_idx
                    
                    for t in freezing_times:
                        ax1.vlines(t, y_pos - 0.4, y_pos + 0.4, color='black', 
                                linewidth=0.5, alpha=0.7)
                        
                    ax1.set_xlabel("Time (s)")
                    ax1.set_ylabel("Animal")
                    ax1.grid(False)
                    ax1.set_yticks([])

                    time_col = animal_data['channels']['time']
                    time_data = animal_data['fiber_cropped'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['channel_data']:
                            target_col = animal_data['channel_data'][channel_num].get(self.target_signal_var.get())
                            if target_col and target_col in animal_data['fiber_cropped'].columns:
                                # Use different colors for different channels
                                colors = ["#48FF00", "#0AC400FF", "#00940F", "#0D6000", "#073B00"]
                                color_idx = (channel_num - 1) % len(colors)
                                color = colors[color_idx]

                                ax2.plot(time_data, animal_data['fiber_cropped'][target_col],
                                    color=color, linewidth=1, 
                                    label=f'{animal_data["animal_id"]} CH{channel_num}')
                
                    ax2.set_xlabel("Time (s)")
                    ax2.set_ylabel("Fiber Signal")
                    ax2.grid(False)
                    ax2.legend()
                    
                    # Add event backgrounds
                    if selected_indices and 'event_data' in self.multi_animal_data[selected_indices[0]]:
                        event_data = self.multi_animal_data[selected_indices[0]]['event_data']
                        video_start = self.multi_animal_data[selected_indices[0]]['video_start_fiber']
                        
                        for _, row in event_data.iterrows():
                            if row['Event Type'] == 0:
                                color = "#b6b6b6"
                            elif row['Event Type'] == 3:
                                color = "#b6b6b6"
                            else:
                                color = "#fcb500"
                                
                            start_time = row['start_time']
                            end_time = row['end_time']
                            
                            ax1.axvspan(start_time, end_time, color=color, alpha=0.3)
                            ax2.axvspan(start_time, end_time, color=color, alpha=0.3)
                            
                    fig.tight_layout()
                    canvas.draw()
            
            self.set_status("Freezing with fiber plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot freezing with fiber: {str(e)}")
            traceback.print_exc()

    def export_freezing_csv(self):
        """Export freezing results for selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Calculate FPS and compute freezing
        self.calculate_fps_from_fiber()
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)

        try:
            all_results = []
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if 'freezing_results' not in animal_data:
                    continue
                
                animal_id = animal_data['animal_id']
                group = animal_data.get('group', 'Unknown')
                
                for result in animal_data['freezing_results']:
                    result_copy = result.copy()
                    result_copy['Animal ID'] = animal_id
                    result_copy['Group'] = group
                    all_results.append(result_copy)
            
            if not all_results:
                messagebox.showinfo("No Data", "No freezing results to export")
                return
            
            df = pd.DataFrame(all_results)
            
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Freezing Results"
            )
            
            if save_path:
                df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Freezing results exported to:\n{save_path}")
                self.set_status(f"Freezing results exported: {len(all_results)} records")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export freezing results: {str(e)}")
            traceback.print_exc()

    def plot_compare_raw_cleaning(self):
        """Compare raw vs cleaned speed data for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        # Calculate FPS and compute freezing
        self.calculate_fps_from_fiber()
        
        for idx in selected_indices:
            animal_data = self.multi_animal_data[idx]
            if 'freezing_results' not in animal_data:
                result = self.compute_freezing_for_animal(animal_data)
                if result:
                    animal_data.update(result)

        try:
            freezing_compute_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "freezing compute":
                    freezing_compute_tab_index = i
                    break
            
            if freezing_compute_tab_index is not None:
                self.plot_notebook.forget(freezing_compute_tab_index)
            
            default_tab_index = None
            for i, tab in enumerate(self.plot_notebook.tabs()):
                tab_text = self.plot_notebook.tab(tab, "text")
                if tab_text == "Plots":
                    default_tab_index = i
                    break
            
            if default_tab_index is not None:
                self.plot_notebook.forget(default_tab_index)
            
            freezing_compute_tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(freezing_compute_tab, text="freezing compute")
            
            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            group_notebook = ttk.Notebook(freezing_compute_tab)
            group_notebook.pack(fill="both", expand=True, padx=5, pady=5)

            # Create subplots for each group
            for group_name, group_animals in groups.items():
                # Create tab for this group
                group_tab = ttk.Frame(group_notebook)
                group_notebook.add(group_tab, text=group_name)
                
                # Create notebook for animals within this group
                animal_notebook = ttk.Notebook(group_tab)
                animal_notebook.pack(fill="both", expand=True, padx=5, pady=5)
                
                for animal_idx, animal_data in enumerate(group_animals):
                    if 'raw_speed' not in animal_data:
                        continue
                    
                    animal_id = animal_data['animal_id']
                    
                    # Create tab for this animal
                    animal_tab = ttk.Frame(animal_notebook)
                    animal_notebook.add(animal_tab, text=animal_id)

                    # Create figure and canvas for this group
                    fig = Figure(figsize=(10, 8), dpi=100)
                    canvas = FigureCanvasTkAgg(fig, master=animal_tab)
                    canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    toolbar_frame = ttk.Frame(animal_tab)
                    toolbar_frame.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                    toolbar.update()
                    
                    # Create subplots
                    ax1 = fig.add_subplot(3, 2, 1)  # dist of le
                    ax2 = fig.add_subplot(3, 2, 3)  # dist of re
                    ax3 = fig.add_subplot(3, 2, 5)  # dist of tb
                    ax4 = fig.add_subplot(3, 2, 2)  # Raw vs interp
                    ax5 = fig.add_subplot(3, 2, 4)  # Interp vs smooth
                    ax6 = fig.add_subplot(3, 2, 6)  # Freezing
                    
                    colors = plt.cm.tab10(np.linspace(0, 1, len(group_animals)))
                    
                    fps = animal_data.get('fps', 30)
                    time_axis = np.arange(len(animal_data['raw_speed'])) / fps
                    
                    ax1.plot(time_axis, animal_data['dist_le'], 
                            color=colors[animal_idx], linewidth=1,
                            label=f"{animal_data['animal_id']} - Left Ear")
                    ax1.plot(time_axis, animal_data['dist_le_i'],
                            color=colors[animal_idx], linewidth=1, linestyle='--',
                            label=f"{animal_data['animal_id']} - Left Ear Interp")
                    
                    ax2.plot(time_axis, animal_data['dist_re'], 
                            color=colors[animal_idx], linewidth=1,
                            label=f"{animal_data['animal_id']} - Right Ear")
                    ax2.plot(time_axis, animal_data['dist_re_i'],
                            color=colors[animal_idx], linewidth=1, linestyle='--',
                            label=f"{animal_data['animal_id']} - Right Ear Interp")
                    
                    ax3.plot(time_axis, animal_data['dist_tb'], 
                            color=colors[animal_idx], linewidth=1,
                            label=f"{animal_data['animal_id']} - Tail Base")
                    ax3.plot(time_axis, animal_data['dist_tb_i'],
                            color=colors[animal_idx], linewidth=1, linestyle='--',
                            label=f"{animal_data['animal_id']} - Tail Base Interp")

                    # Plot raw vs interpolated
                    ax4.plot(time_axis, animal_data['raw_speed'], 
                            color=colors[animal_idx], linewidth=1, alpha=0.7,
                            label=f"{animal_data['animal_id']} - Raw")
                    ax4.plot(time_axis, animal_data['interp_speed'], 
                            color=colors[animal_idx], linewidth=1.5, linestyle='--',
                            label=f"{animal_data['animal_id']} - Interpolated")
                    
                    # Plot interpolated vs smoothed
                    ax5.plot(time_axis, animal_data['interp_speed'], 
                            color=colors[animal_idx], linewidth=1, alpha=0.7,
                            label=f"{animal_data['animal_id']} - Interpolated")
                    ax5.plot(time_axis, animal_data['smooth_speed'], 
                            color=colors[animal_idx], linewidth=1.5, linestyle='--',
                            label=f"{animal_data['animal_id']} - Smoothed")

                    ax6.plot(time_axis, animal_data['smooth_speed'], 
                            color=colors[animal_idx], linewidth=1,
                            label=f"{animal_data['animal_id']} - Speed")
                    
                    freeze_mask = np.array(animal_data['dlc_freeze_status'], dtype=bool)
                    if np.any(freeze_mask):
                        ax6.fill_between(time_axis, 0, animal_data['smooth_speed'].max() * 1.1,
                                       where=freeze_mask, alpha=0.3, color=colors[animal_idx])
                    
                    ax6.axhline(y=animal_data.get('freeze_threshold', 1.2), color='black', linestyle='--',
                            label='Freezing Threshold')
                    
                    ax1.set_ylabel("Speed (pixels/frame)")
                    ax1.set_title(f"Raw vs Interpolated Speed of Left Ear - Group {group_name}")
                    ax1.grid(False)
                    ax1.legend()

                    ax2.set_ylabel("Speed (pixels/frame)")
                    ax2.set_title(f"Raw vs Interpolated Speed of Right Ear - Group {group_name}")
                    ax2.grid(False)
                    ax2.legend()

                    ax3.set_xlabel("Time (s)")
                    ax3.set_ylabel("Speed (pixels/frame)")
                    ax3.set_title(f"Raw vs Interpolated Speed of Tail Base - Group {group_name}")
                    ax3.grid(False)
                    ax3.legend()

                    ax4.set_ylabel("Speed (pixels/frame)")
                    ax4.set_title(f"Raw vs Interpolated Speed - Group {group_name}")
                    ax4.grid(False)
                    ax4.legend()
                    
                    ax5.set_ylabel("Speed (pixels/frame)")
                    ax5.set_title(f"Interpolated vs Smoothed Speed - Group {group_name}")
                    ax5.grid(False)
                    ax5.legend()

                    ax6.set_xlabel("Time (s)")
                    ax6.set_ylabel("Smoothed Speed (pixels/frame)")
                    ax6.set_title(f"Freezing with Fiber - Group {group_name}")
                    ax6.grid(False)
                    ax6.legend()
                
                    fig.tight_layout()
                    canvas.draw()
            
            self.set_status("Raw vs cleaned comparison plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot comparison: {str(e)}")
            traceback.print_exc()

    def h_AST2_readData(self, file_path):
        """Read AST2 data file"""
        try:
            with open(file_path, 'rb') as fileID:
                fileID.seek(0, 0)
                version = struct.unpack('>H', fileID.read(2))[0]
                headerSize = struct.unpack('>H', fileID.read(2))[0]
                nChannels = struct.unpack('>H', fileID.read(2))[0]
                nScans = struct.unpack('>I', fileID.read(4))[0]
                scanRate = struct.unpack('>H', fileID.read(2))[0]
                
                fileID.seek(headerSize, 0)
                data = np.fromfile(fileID, dtype='>i2')
                data = np.reshape(data, (nScans, nChannels))
                
            return data
        except Exception as e:
            print(f"Error reading AST2 file: {str(e)}")
            return None

class PupilAnalyzerApp(BaseAnalyzerApp):
    def __init__(self, root, include_fiber=False):
        super().__init__(root, include_fiber)
        self.root.title("ML Lab Pupil Analyzer")
        self.root.state('zoomed')
        self.create_ui()

    def create_main_controls(self):
        control_panel = ttk.Frame(self.control_frame)
        control_panel.pack(fill="x", padx=5, pady=5)

        # Multi-animal list frame
        multi_animal_frame = ttk.LabelFrame(control_panel, text="Multi Animal Analysis")
        multi_animal_frame.pack(fill="x", padx=5, pady=5)
        
        list_frame = ttk.Frame(multi_animal_frame)
        list_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, height=5)
        self.file_listbox.pack(fill=tk.X, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        btn_frame = ttk.Frame(multi_animal_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="Single Import", command=self.single_import).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Clear Selected", command=self.clear_selected).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Multi Import", command=self.multi_import).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="Select All", command=self.select_all).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        
        # Pupil analysis buttons
        pupil_frame = ttk.LabelFrame(control_panel, text="Pupil Analysis")
        pupil_frame.pack(fill="x", padx=5, pady=5)
        button_frame = ttk.Frame(pupil_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        ttk.Button(button_frame, text="Plot Pupil Distance", command=self.plot_pupil_distance).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Plot Combined Data", command=self.plot_combined_data).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Export Pupil", command=self.export_pupil_csv).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        
        if self.include_fiber:
            fiber_frame = ttk.LabelFrame(control_panel, text="Fiber Photometry")
            fiber_frame.pack(fill="x", padx=5, pady=5)
            fiber_frame.columnconfigure(0, weight=1)
            fiber_frame.columnconfigure(1, weight=1)
            
            ttk.Button(fiber_frame, text="Plot Raw Data", command=self.plot_raw_data).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
            ttk.Button(fiber_frame, text="Plot Pupil w/Fiber", command=self.plot_pupil_with_fiber).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
    
    def show_step1(self):
        self.set_status("Step 1: Data Loading")
    
    def show_step2(self):
        self.set_status("Step 2: Preprocessing")

        if not hasattr(self, 'step2_frame') or not self.step2_frame:
            self.step2_frame = ttk.LabelFrame(self.control_frame, text="Preprocessing")
            self.step2_frame.pack(fill="x", padx=5, pady=5)
        
        self.step2_frame.pack(fill="x", padx=5, pady=5)
        self.step2_frame.update_idletasks()
        
        for widget in self.step2_frame.winfo_children():
            widget.destroy()
        
        signal_frame = ttk.Frame(self.step2_frame)
        signal_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(signal_frame, text="Target Signal:").grid(row=0, column=0, sticky="w")
        target_options = ["470", "560"]
        target_menu = ttk.OptionMenu(signal_frame, self.target_signal_var, "470", *target_options)
        target_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        ttk.Label(signal_frame, text="Reference Signal:").grid(row=1, column=0, sticky="w")
        ref_options = ["410", "baseline"]
        ref_menu = ttk.OptionMenu(signal_frame, self.reference_signal_var, "410", *ref_options)
        ref_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        self.baseline_frame = ttk.Frame(self.step2_frame)
        self.baseline_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(self.baseline_frame, text="Baseline Period (s):").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.baseline_frame, textvariable=self.baseline_start, width=6).grid(row=0, column=1, padx=2)
        ttk.Label(self.baseline_frame, text="to").grid(row=0, column=2)
        ttk.Entry(self.baseline_frame, textvariable=self.baseline_end, width=6).grid(row=0, column=3, padx=2)
        
        if self.reference_signal_var.get() != "baseline":
            self.baseline_frame.pack_forget()
        
        smooth_frame = ttk.Frame(self.step2_frame)
        smooth_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(smooth_frame, text="Apply Smoothing", variable=self.apply_smooth,
                       command=lambda: self.toggle_widgets(smooth_frame, self.apply_smooth.get(), 1)).grid(row=0, column=0, sticky="w")
        
        param_frame = ttk.Frame(smooth_frame)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Label(param_frame, text="Window Size:").grid(row=0, column=0, sticky="w")
        ttk.Scale(param_frame, from_=3, to=101, orient=tk.HORIZONTAL, 
                 length=100, variable=self.smooth_window,
                 command=lambda v: self.smooth_window.set(int(float(v)))).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_window).grid(row=0, column=2)
        
        ttk.Label(param_frame, text="Polynomial Order:").grid(row=1, column=0, sticky="w")
        ttk.Scale(param_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                 length=100, variable=self.smooth_order,
                 command=lambda v: self.smooth_order.set(int(float(v)))).grid(row=1, column=1, padx=5)
        ttk.Label(param_frame, textvariable=self.smooth_order).grid(row=1, column=2)
        
        param_frame.grid_remove()
        
        baseline_corr_frame = ttk.Frame(self.step2_frame)
        baseline_corr_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(baseline_corr_frame, text="Apply Baseline Correction", variable=self.apply_baseline,
                        command=lambda: self.toggle_widgets(baseline_corr_frame, self.apply_baseline.get(), 1)).grid(row=0, column=0, sticky="w")
        
        model_frame = ttk.Frame(baseline_corr_frame)
        model_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        ttk.Label(model_frame, text="Baseline Model:").grid(row=1, column=0, sticky="w")
        model_options = ["Polynomial", "Exponential"]
        model_menu = ttk.OptionMenu(model_frame, self.baseline_model, "Polynomial", *model_options)
        model_menu.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        model_frame.grid_remove()
        
        motion_frame = ttk.Frame(self.step2_frame)
        motion_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(motion_frame, text="Apply Motion Correction", variable=self.apply_motion,
                       command=lambda: self.toggle_motion_correction()).grid(row=0, column=0, sticky="w")
        
        ttk.Button(self.step2_frame, text="Apply Preprocessing", 
                  command=self.apply_preprocessing).pack(pady=10)
        
        self.reference_signal_var.trace_add("write", self.update_baseline_ui)

    def update_baseline_ui(self, *args):
        if self.reference_signal_var.get() == "baseline":
            self.baseline_frame.pack(fill="x", padx=5, pady=5)
        else:
            self.baseline_frame.pack_forget()
    
    def toggle_motion_correction(self):
        if self.reference_signal_var.get() != "410":
            self.apply_motion.set(False)
            messagebox.showinfo("Info", "Motion correction requires 410nm as reference signal")
    
    def toggle_widgets(self, parent_frame, show, index):
        children = parent_frame.winfo_children()
        if len(children) > index:
            if show:
                children[index].grid()
            else:
                children[index].grid_remove()
    
    def show_step3(self):
        self.set_status("Step 3: ΔF/F & Z-score")
        
        if not hasattr(self, 'step3_frame') or not self.step3_frame:
            self.step3_frame = ttk.LabelFrame(self.control_frame, text="ΔF/F & Z-score")
            self.step3_frame.pack(fill="x", padx=5, pady=5)
        
        self.step3_frame.pack(fill="x", padx=5, pady=5)
        self.step3_frame.update_idletasks()
        
        for widget in self.step3_frame.winfo_children():
            widget.destroy()

        self.step3_frame.columnconfigure(0, weight=1)
        self.step3_frame.columnconfigure(1, weight=1)
        
        ttk.Button(self.step3_frame, text="Calculate ΔF/F", command=self.calculate_and_plot_dff).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(self.step3_frame, text="Calculate Z-score", command=self.calculate_and_plot_zscore).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
    
    def show_step4(self):
        self.set_status("Step 4: Event Analysis")

        if not hasattr(self, 'step4_frame') or not self.step4_frame:
            self.step4_frame = ttk.LabelFrame(self.control_frame, text="Event Analysis")
            self.step4_frame.pack(fill="x", padx=5, pady=5)
        
        self.step4_frame.pack(fill="x", padx=5, pady=5)
        self.step4_frame.update_idletasks()
        
        for widget in self.step4_frame.winfo_children():
            widget.destroy()
        
        btn_frame = ttk.Frame(self.step4_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(btn_frame, text="Plot Event-Activity & Heatmap", 
                  command=self.show_event_activity_dialog).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Plot Experiment-Activity", 
                  command=self.show_experiment_activity_dialog).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        ttk.Button(btn_frame, text="Export Statistics", 
                  command=self.export_statistic_results).grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

    def plot_pupil_distance(self):
        """Plot pupil distance for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            # Clear existing tabs
            for tab in self.plot_notebook.tabs():
                self.plot_notebook.forget(tab)
            
            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            # Create subplots for each group
            for group_name, group_animals in groups.items():
                # Create tab for this group
                group_tab = ttk.Frame(self.plot_notebook)
                self.plot_notebook.add(group_tab, text=group_name)
                
                # Create figure and canvas for this group
                fig = Figure(figsize=(10, 6), dpi=100)
                canvas = FigureCanvasTkAgg(fig, master=group_tab)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                
                toolbar_frame = ttk.Frame(group_tab)
                toolbar_frame.pack(fill="x")
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                
                ax = fig.add_subplot(111)
                
                for animal_data in group_animals:
                    if 'dlc_data' not in animal_data:
                        continue
                    
                    dlc_result = animal_data['dlc_data']
                    scorer = dlc_result.columns.levels[0][0]
                    
                    # Calculate pupil distance
                    left_pupil_x = dlc_result.loc[:, (scorer, 'left_pupil', 'x')].values.astype(float)
                    left_pupil_y = dlc_result.loc[:, (scorer, 'left_pupil', 'y')].values.astype(float)
                    right_pupil_x = dlc_result.loc[:, (scorer, 'right_pupil', 'x')].values.astype(float)
                    right_pupil_y = dlc_result.loc[:, (scorer, 'right_pupil', 'y')].values.astype(float)
                    
                    pupil_distance = np.sqrt((left_pupil_x - right_pupil_x)**2 + 
                                           (left_pupil_y - right_pupil_y)**2)
                    
                    time_axis = np.arange(len(pupil_distance)) / 30  # Assuming 30 FPS
                    
                    ax.plot(time_axis, pupil_distance, 
                           label=f"{animal_data['animal_id']} - Pupil Distance")
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Pupil Distance (pixels)")
                ax.set_title(f"Pupil Distance - Group {group_name}")
                ax.grid(False)
                ax.legend()
                
                canvas.draw()
            
            self.set_status("Pupil distance plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot pupil distance: {str(e)}")
            traceback.print_exc()

    def plot_combined_data(self):
        """Plot combined data for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            # Clear existing tabs
            for tab in self.plot_notebook.tabs():
                self.plot_notebook.forget(tab)
            
            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            # Create subplots for each group
            for group_name, group_animals in groups.items():
                # Create tab for this group
                group_tab = ttk.Frame(self.plot_notebook)
                self.plot_notebook.add(group_tab, text=group_name)
                
                # Create figure and canvas for this group
                fig = Figure(figsize=(10, 8), dpi=100)
                canvas = FigureCanvasTkAgg(fig, master=group_tab)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                
                toolbar_frame = ttk.Frame(group_tab)
                toolbar_frame.pack(fill="x")
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                
                # Create subplots
                ax1 = fig.add_subplot(2, 1, 1)  # Pupil data
                ax2 = fig.add_subplot(2, 1, 2)  # Fiber data
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(group_animals)))
                
                # Plot pupil data
                for animal_idx, animal_data in enumerate(group_animals):
                    if 'dlc_data' not in animal_data:
                        continue
                    
                    dlc_result = animal_data['dlc_data']
                    scorer = dlc_result.columns.levels[0][0]
                    
                    left_pupil_x = dlc_result.loc[:, (scorer, 'left_pupil', 'x')].values.astype(float)
                    left_pupil_y = dlc_result.loc[:, (scorer, 'left_pupil', 'y')].values.astype(float)
                    right_pupil_x = dlc_result.loc[:, (scorer, 'right_pupil', 'x')].values.astype(float)
                    right_pupil_y = dlc_result.loc[:, (scorer, 'right_pupil', 'y')].values.astype(float)
                    
                    pupil_distance = np.sqrt((left_pupil_x - right_pupil_x)**2 + 
                                           (left_pupil_y - right_pupil_y)**2)
                    
                    time_axis = np.arange(len(pupil_distance)) / 30
                    
                    ax1.plot(time_axis, pupil_distance, 
                            color=colors[animal_idx], linewidth=1,
                            label=f"{animal_data['animal_id']} - Pupil Distance")
                
                ax1.set_ylabel("Pupil Distance (pixels)")
                ax1.set_title(f"Combined Data - Group {group_name}")
                ax1.grid(False)
                ax1.legend()
                
                # Plot fiber data
                for animal_idx, animal_data in enumerate(group_animals):
                    if 'dff_data' not in animal_data:
                        continue
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['dff_data']:
                            dff_data = animal_data['dff_data'][channel_num]
                            ax2.plot(time_data, dff_data, 
                                    color=colors[animal_idx], linewidth=1, alpha=0.7,
                                    label=f"{animal_data['animal_id']} CH{channel_num} dF/F")
                
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("ΔF/F")
                ax2.grid(False)
                ax2.legend()
                
                fig.tight_layout()
                canvas.draw()
            
            self.set_status("Combined data plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot combined data: {str(e)}")
            traceback.print_exc()

    def plot_pupil_with_fiber(self):
        """Plot pupil with fiber data for selected animals with group-based visualization"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            # Clear existing tabs
            for tab in self.plot_notebook.tabs():
                self.plot_notebook.forget(tab)
            
            # Group animals by group
            groups = {}
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                group = animal_data.get('group', 'Unknown')
                if group not in groups:
                    groups[group] = []
                groups[group].append(animal_data)
            
            # Create subplots for each group
            for group_name, group_animals in groups.items():
                # Create tab for this group
                group_tab = ttk.Frame(self.plot_notebook)
                self.plot_notebook.add(group_tab, text=group_name)
                
                # Create figure and canvas for this group
                fig = Figure(figsize=(10, 8), dpi=100)
                canvas = FigureCanvasTkAgg(fig, master=group_tab)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                
                toolbar_frame = ttk.Frame(group_tab)
                toolbar_frame.pack(fill="x")
                toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                toolbar.update()
                
                # Create subplots
                ax1 = fig.add_subplot(2, 1, 1)  # Pupil
                ax2 = fig.add_subplot(2, 1, 2)  # Fiber data
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(group_animals)))
                
                # Plot pupil data
                for animal_idx, animal_data in enumerate(group_animals):
                    if 'dlc_data' not in animal_data:
                        continue
                    
                    dlc_result = animal_data['dlc_data']
                    scorer = dlc_result.columns.levels[0][0]
                    
                    left_pupil_x = dlc_result.loc[:, (scorer, 'left_pupil', 'x')].values.astype(float)
                    left_pupil_y = dlc_result.loc[:, (scorer, 'left_pupil', 'y')].values.astype(float)
                    right_pupil_x = dlc_result.loc[:, (scorer, 'right_pupil', 'x')].values.astype(float)
                    right_pupil_y = dlc_result.loc[:, (scorer, 'right_pupil', 'y')].values.astype(float)
                    
                    pupil_distance = np.sqrt((left_pupil_x - right_pupil_x)**2 + 
                                           (left_pupil_y - right_pupil_y)**2)
                    
                    time_axis = np.arange(len(pupil_distance)) / 30
                    
                    ax1.plot(time_axis, pupil_distance, 
                            color=colors[animal_idx], linewidth=1,
                            label=f"{animal_data['animal_id']} - Pupil Distance")
                
                ax1.set_ylabel("Pupil Distance (pixels)")
                ax1.set_title(f"Pupil with Fiber - Group {group_name}")
                ax1.grid(False)
                ax1.legend()
                
                # Plot fiber data
                for animal_idx, animal_data in enumerate(group_animals):
                    if 'dff_data' not in animal_data:
                        continue
                    
                    time_col = animal_data['channels']['time']
                    time_data = animal_data['preprocessed_data'][time_col] - animal_data['video_start_fiber']
                    
                    for channel_num in animal_data['active_channels']:
                        if channel_num in animal_data['dff_data']:
                            dff_data = animal_data['dff_data'][channel_num]
                            ax2.plot(time_data, dff_data, 
                                    color=colors[animal_idx], linewidth=1, alpha=0.7,
                                    label=f"{animal_data['animal_id']} CH{channel_num} dF/F")
                
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("ΔF/F")
                ax2.grid(False)
                ax2.legend()
                
                fig.tight_layout()
                canvas.draw()
            
            self.set_status("Pupil with fiber plotted")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot pupil with fiber: {str(e)}")
            traceback.print_exc()

    def export_pupil_csv(self):
        """Export pupil results for selected animals"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select animals from the list")
            return
        
        try:
            all_results = []
            
            for idx in selected_indices:
                animal_data = self.multi_animal_data[idx]
                if 'dlc_data' not in animal_data:
                    continue
                
                animal_id = animal_data['animal_id']
                group = animal_data.get('group', 'Unknown')
                
                dlc_result = animal_data['dlc_data']
                scorer = dlc_result.columns.levels[0][0]
                
                left_pupil_x = dlc_result.loc[:, (scorer, 'left_pupil', 'x')].values.astype(float)
                left_pupil_y = dlc_result.loc[:, (scorer, 'left_pupil', 'y')].values.astype(float)
                right_pupil_x = dlc_result.loc[:, (scorer, 'right_pupil', 'x')].values.astype(float)
                right_pupil_y = dlc_result.loc[:, (scorer, 'right_pupil', 'y')].values.astype(float)
                
                pupil_distance = np.sqrt((left_pupil_x - right_pupil_x)**2 + 
                                       (left_pupil_y - right_pupil_y)**2)
                
                time_axis = np.arange(len(pupil_distance)) / 30
                
                for i, (time_val, dist_val) in enumerate(zip(time_axis, pupil_distance)):
                    all_results.append({
                        'Group': group,
                        'Animal ID': animal_id,
                        'Time (s)': time_val,
                        'Pupil Distance (pixels)': dist_val,
                        'Frame': i
                    })
            
            if not all_results:
                messagebox.showinfo("No Data", "No pupil data to export")
                return
            
            df = pd.DataFrame(all_results)
            
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Pupil Results"
            )
            
            if save_path:
                df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Pupil results exported to:\n{save_path}")
                self.set_status(f"Pupil results exported: {len(all_results)} records")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export pupil results: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModeSelectionApp(root)
    root.mainloop()