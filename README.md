# openvino-tests

Repo to test some Intel OpenVINO pretrained models.

## Prerequisites:

* Python 3 (version >= 3.5)
* Download and install [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html). This repo was tested with OpenVINO version 2019.R3.
* Be sure to source OpenVINO env (could be different depending on your OS):
```
source /opt/intel/openvino/bin/setupvars.sh
```
* Create and activate a Python virtual environment:
```
python3 -m venv <some-name>
source <some-name>/bin/activate
```
* Install requirements:
```
pip install -r requirements.txt
```
* Run the demo:
```
python demo_app.py
```

TODO:
* Change code to run demo_app passing arguments via command line.