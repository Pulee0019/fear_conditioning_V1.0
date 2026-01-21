# 1 Driver Installation
## 1.1 NI card driver
### 1.1.1 **NI-DAQmx**: [NI-DAQmx official website](https://www.ni.com/zh-cn/support/downloads/drivers/download.ni-daq-mx.html#569353)
![NI-DAQmx](https://github.com/Pulee0019/fear_conditioning_V1.0/blob/main/imgs/NI-DAQmx.png)
## 1.2 FLIR camera driver
### 1.2.1 **PySpin 4.3 for Windows (Windows 64-bit -- Python 3.10)**:  [FlIR camera driver official website](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK)
![FLIR1](https://github.com/Pulee0019/fear_conditioning_V1.0/blob/main/imgs/FLIR1.png)
### 1.2.2 **Spinnaker SDK 4.3 for Windows (Full Version) (Windows 64-bit -- Full)**:  [FLIR camera driver offical website](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK)
![FLIR2](https://github.com/Pulee0019/fear_conditioning_V1.0/blob/main/imgs/FLIR2.png)
> **The download speed maybe slow.**
## 1.3 HIKI robot camera driver
### 1.3.1 MVS: [HIKIrobot camera driver official website](https://www.hikrobotics.com/cn/machinevision/service/download/?module=0)
![HIKI robot](https://github.com/Pulee0019/fear_conditioning_V1.0/blob/main/imgs/HIKI%20robot.png)
> After installation, find and edit **MvCameraControl_class. py**, change the dllname = r"your download path\fear training\lib\MvCameraControl.dll" to your **MvCameraControl.dll** path, save it.
![MvCameraControl](https://github.com/Pulee0019/fear_conditioning_V1.0/blob/main/imgs/MvCameraControl.png)
![code](https://github.com/Pulee0019/fear_conditioning_V1.0/blob/main/imgs/code.png)
![MvCameraControlGUI](https://github.com/Pulee0019/fear_conditioning_V1.0/blob/main/imgs/MvCameraControlGUI.png)
# 2 Configure environment in Anaconda Prompt
## 2.1 Create and activate environment
```
conda create -n fear_training python=3.10 -y    
conda activate fear_training
```
## 2.2 Installation
```
python -m ensurepip
python -m pip install --upgrade pip numpy matplotlib
pip install pyserial
pip install sounddevice
pip install opencv-python
pip intsall opencv-contrib-python
pip install PyDAQmx
pip install pywin32
pip install pandas
pip install spicy
```
## 2.3 Install the spinnaker to environment
> Change the working path to your file path directory of **spinnaker_python-4.2.0.83-cp310-cp310-win_amd64.whl** (in .zip of FLIR camera driver for python 3.10 windows)
![spinnaker](https://github.com/Pulee0019/fear_conditioning_V1.0/blob/main/imgs/spinnaker.png)
### 2.3.1 Change the path, if your spinnaker path in C driver
`cd your spinnaker path`
### 2.3.2 Change the path, if your spinnaker path in other driver
`cd /d your spinnaker path`
### 2.3.3 Install spinnaker
`python -m pip install spinnaker_python-4.2.0.83-cp310-cp310-win_amd64.whl`