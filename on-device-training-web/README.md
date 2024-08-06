# Steps on running ```on-device-training-web``` demo

Clone this repository / branch to proceed with running the demo
## 1. Python

### Initial Setup
Install dependencies. This step requires onnxruntime-training-cpu>=1.17.0. 
```
pip install -r requirements.txt
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu
```

### Generate artifacts

Run the cells in the Jupyter notebook to generate the artifacts, this is required for the web step. You can generate a forward only onnx model, here torch.export is used. 

> For the purpose of this demo, the artifacts have been generated and can be found in the path: ```on-device-training-web/web/public```

## 2. Web
Once the repository is download, open it in terminal or the IDE of choice. Assuming the primary branch is ```sonar```, execute the following steps
```
cd on-device-training-web
cd web
npm install
npm start
```
Ensure you have React and onnx-runtime downloaded. If you do not have the onnx-runtime package downloaded, follow the steps here: 

INSTALL ONNX RUNTIME WEB (BROWSERS)
```
# install latest release version
npm install onnxruntime-web

# install nightly build dev version
npm install onnxruntime-web@dev
```
INSTALL ONNX RUNTIME NODE.JS BINDING (NODE.JS)
```
# install latest release version
npm install onnxruntime-node
```

INSTALL ONNX RUNTIME FOR REACT NATIVE
```
# install latest release version
npm install onnxruntime-react-native
```

## Additional Info
This POC is based on: https://github.com/microsoft/onnxruntime-training-examples/tree/05c70f78b824bbdd629cb84cff17aebfcc786956/on_device_training/web. 
This has been adapted for the xsum dataset. Going forward, this can serve as an example for training other models on other datasets.
