## Install onnx、onnxruntime
if you have a gpu machine run
```
pip install onnx onnxruntime-gpu
```
or you only have cpu machine run below
```
pip install onnx onnxruntime
```

## Download weights
```
huggingface-cli download tzhhhh/sv4pdd-dwpose --local-dir ckpts
```
## ps
dwpose code is borrowed from [dwpose-onnx](https://github.com/IDEA-Research/DWPose)