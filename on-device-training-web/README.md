This POC is based on: https://github.com/microsoft/onnxruntime-training-examples/tree/05c70f78b824bbdd629cb84cff17aebfcc786956/on_device_training/web --> adapted for the xsum dataset, this can be modified going forward

# Offline step
This step is the same regardless of the method used to import the ONNXRuntime-web/training package. We are opting for web. 

## Set-up

Install dependencies. This step requires onnxruntime-training-cpu>=1.17.0. 
```
pip install -r requirements.txt
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu
```

## Generate artifacts

Run the cells in the Jupyter notebook to generate the artifacts, this is required for the web step. You can generate a forward only onnx model, here torch.export is used. 