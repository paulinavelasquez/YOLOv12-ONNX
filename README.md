# YOLOv12 ONNX Python

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/language-Python-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-v1.17.0-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-brightgreen.svg)

## Overview

This repository provides a clean and modular pipeline to **export YOLOv12 models from PyTorch (.pt) to ONNX**, and then perform **fast inference with ONNX Runtime in Python**.  
Using ONNX offers several benefits:
- **Cross-platform compatibility** (deploy the same model on Python, C++, Java, or mobile runtimes).
- **High performance** inference with [ONNX Runtime](https://onnxruntime.ai/) on both CPU and GPU.
- **Lightweight deployment** without requiring the entire PyTorch framework.
- **Easier integration** into production environments.

This project includes:
- Scripts to export YOLOv12 `.pt` weights to `.onnx`.
- A modular `YOLOONNXPredictor` class for inference.
- Utilities to draw bounding boxes with class names and confidence scores.

<p align="center">
  <img src="data/example_output.jpg" width=500>
</p>

---

## Output Example

Below is an example of running detection with YOLOv12 ONNX on an input image:

<div align="center">
  <h3>Image Inference Output</h3>
  <img src="data/output_example.jpg" alt="Image Output" width="500">
</div>

---

## Features

- **ONNX Runtime Integration**: Fast inference with CPU or GPU backends.
- **Dynamic Shapes**: Optional support for variable input sizes.
- **Post-processing Included**: Handles confidence thresholding and NMS.
- **Custom Class Labels**: Works with custom datasets (not only COCO).
- **Visualization Utilities**: Draw bounding boxes and labels on images.

---

## Requirements

Install the following dependencies inside your Python environment:

```bash
pip install ultralytics onnx onnxruntime opencv-python pillow numpy
