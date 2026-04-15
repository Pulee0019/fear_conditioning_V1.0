from pathlib import Path
from track2p.t2p import run_t2p                     # main function that launches track2p
from track2p.ops.default import DefaultTrackOps     # default track2p options
import pandas as pd
import os


# load default settings / parameters
track_ops = DefaultTrackOps()
track_ops.reg_chan = 0
track_ops.iscell_thr=None
track_ops.save_in_s2p_format = True
track_ops.iou_dist_thr= 10

current_dir = Path(os.getcwd())
groups_files_dir = current_dir / "2PCIFA" / "groups"
groups_files_paths = [os.path.join(groups_files_dir, f) for f in os.listdir(groups_files_dir)]
for group_file in groups_files_paths:
    group_name = os.path.basename(group_file).split('.')[0]
    save_path = current_dir / "2PCIFA" / "data" / "data_t2p" / group_name
    os.makedirs(save_path, exist_ok=True)
    group_df = pd.read_csv(group_file)
    group_list = group_df.values.tolist()
    for i, group in enumerate(group_list):
        track_ops.all_ds_path = group
        track_ops.save_path = os.path.join(save_path, f'paired{i+1}')
        for attr, value in track_ops.__dict__.items():
            print(attr, '=', value)
        run_t2p(track_ops)