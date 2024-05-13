#!/bin/bash

# Rebuild the model chunks

# YOLO MODEL
cat asset/model/yolo/yolov8l_chunks/pt/* > asset/model/yolo/yolov8l.pt
cat asset/model/yolo/yolov8l_chunks/onnx/* > asset/model/yolo/yolov8l.onnx