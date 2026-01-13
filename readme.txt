Install NI card driver: NI-DAQmx(official website: https://www.ni.com/zh-cn/support/downloads/drivers/download.ni-daq-mx.html#569353)
Install FLIR camera driver for python 3.10 windows(.zip with .whl for pyspin) and full(.zip with .exe for application)(official website: https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK)  Note: You need connect VPN because the download speed is very slow.
Install HIKI robot camera driver(official website: https://www.hikrobotics.com/cn/machinevision/service/download/?module=0)
Open the MvCameraControl_class, change the dllname = r"D:\Expriment\Code\Python\fear training\lib\MvCameraControl.dll" to your "MvCameraControl.dll" path, save it.

conda create -n fear_training python=3.10 -y
conda activate fear_training
pip config set install.trusted-host mirrors.aliyun.com
python -m ensurepip
python -m pip install --upgrade pip numpy matplotlib
cd /d D:\APP\Spinnaker(your file path directory of spinnaker_python-4.2.0.83-cp310-cp310-win_amd64.whl(in .zip of FLIR camera driver for python 3.10 windows))
python -m pip install spinnaker_python-4.2.0.83-cp310-cp310-win_amd64.whl
pip install pyserial
pip install sounddevice
pip install opencv-python
pip intsall opencv-contrib-python
pip install PyDAQmx
pip install pywin32
pip install pandas
pip install spicy