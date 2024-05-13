#!/bin/bash

pip install -r requirements.txt

# Your machine has CUDA9 or CUDA10 installed, run the following command to install it
# I'm running on CUDA 12.x is OK
# There is also CPU-ONLY this is the documentation for more details
# https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/PULC/PULC_vehicle_attribute.md
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

# onnxruntime-gpu for CUDA version 12.x
# You can refer to this documentation for installing onnxruntime-gpu
# https://onnxruntime.ai/docs/install/#install-onnx-runtime-ort-1
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install onnx

