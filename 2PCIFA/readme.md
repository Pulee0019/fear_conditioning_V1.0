# 1 Environment Requirement
## 1.1 Environment Create and Activate
```
conda create -n suite2p_track2p python=3.9 -y
conda activate suite2p_track2p
```
## 1.2 Installation
```
python -m pip install suite2p[gui]
pip install track2p
pip install h5py
```
# 2 Code User Manual
## 2.1 Run batch suite2p
### 2.1.1 Run_batch_suite2p.py
> Run this code, you should change the `tif_files_path = r"F:\Calcium-fear"` in function `main` to **your root include all needed precessed tif files**.
## 2.2 Run batch track2p
### 2.2.1 Paired the session
> Find the path include suite2p file and save it to `d1.csv` and `a2a.csv`, which locating in file named **group** in your **code path**.
### 2.2.2 Run_batch_track2p.py
> Run this code.
## 2.3 Run main analysis
### 2.3.1 main.ipynb
> Run this code