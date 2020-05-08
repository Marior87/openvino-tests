# openvino-tests

Repo to test some Intel OpenVINO pretrained models.

## Prerequisites:

* Python 3 (version >= 3.5)
* Download and install [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html). This repo was tested with OpenVINO version 2019.R3.
* Be sure to source OpenVINO env (could be different depending on your OS):
```
source /opt/intel/openvino/bin/setupvars.sh
```
* Clone this repo:
```
git clone https://github.com/Marior87/openvino-tests.git
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
* Run the demo (webcam):
```
python demo_app.py -i CAM
```

### Other information:
This repo comes with downloaded pretrained models from [Intel](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models). But you can opt to use different ones, please note that you would likely need to make some code changes for using them.

In addition, OpenVINO offers ways to convert models from popular frameworks like TensorFlow, PyTorch, Cafee, etc. Please check related [documentation](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model.html).