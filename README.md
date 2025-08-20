# YOLOv12 ONNX Python

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/language-Python-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-v1.17.0-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-brightgreen.svg)

## Overview

This repository provides a **clean and modular pipeline** to export **YOLOv12 models from PyTorch (.pt) to ONNX**, and then perform **fast inference with ONNX Runtime in Python**.  

Using ONNX offers several benefits:

- **Cross-platform compatibility**: Deploy the same model on Python, C++, Java, or even mobile runtimes.  
- **High performance**: Optimized inference with [ONNX Runtime](https://onnxruntime.ai/) on both CPU and GPU.  
- **Lightweight deployment**: No need for the full PyTorch framework at inference time.  
- **Production-ready integration**: Easier to embed into applications and services.  

This project includes:

- A script to export YOLOv12 `.pt` weights to `.onnx`.  
- A modular `YOLOONNXPredictor` class for inference.  
- A utility function to draw bounding boxes and labels.  
- An example notebook `ONNX.ipynb` with structured code and usage examples.  

<p align="center">
  <img src="data/example_output.jpg" width=500>
</p>

---

## Output Example

Example of running YOLOv12 ONNX inference on an image:

<div align="center">
  <h3>Image Inference Output</h3>
  <img src="data/output_example.jpg" alt="Image Output" width="500">
</div>

---

## Features

- **ONNX Runtime Integration**: Inference with CPU or GPU backends.  
- **Dynamic Shapes**: Optional support for variable input sizes.  
- **Post-processing Included**: Confidence thresholding and NMS integrated.  
- **Custom Dataset Support**: Works with your own trained classes, not only COCO.  
- **Visualization Utilities**: Bounding boxes and labels drawn on images.  
- **Notebook Example**: The complete structured workflow is demonstrated in `ONNX.ipynb`.  

---

## Requirements

Install the dependencies inside your Python environment:

```bash
pip install ultralytics onnx onnxruntime opencv-python pillow numpy
```

Optional (to simplify exported models):

```bash
pip install onnxsim
```

---

## Exporting YOLOv12 to ONNX

You can export your `.pt` model to ONNX with:

```bash
python export_yolo_to_onnx.py --weights best.pt --imgsz 640 640 --opset 12 --dynamic --simplify
```

Arguments:
- `--weights`: Path to your `.pt` model.  
- `--imgsz`: Input size (H W).  
- `--opset`: ONNX opset version (default: 12).  
- `--dynamic`: Enable dynamic input shapes.  
- `--simplify`: Simplify the exported graph.  

---

## Inference with ONNX

The repository provides a modular class `YOLOONNXPredictor` for inference.

### Example Usage

```python
from yolo_onnx_infer import YOLOONNXPredictor

# class names from your dataset
map_class_names = = model.names

# load ONNX model
predictor = YOLOONNXPredictor("best.onnx", providers=["CPUExecutionProvider"], imgsz=640, class_names=map_class_names)

# run inference
results = predictor.predict("test.jpg", conf_thres=0.25, iou_thres=0.5)
print(results[:5])
```

Output format:

```python
[
 {'x': 0.52, 'y': 0.41, 'w': 0.12, 'h': 0.08, 'conf': 0.91, 'cls': 0, 'label': 'dog'},
 {'x': 0.35, 'y': 0.66, 'w': 0.20, 'h': 0.15, 'conf': 0.85, 'cls': 2, 'label': 'Mascot'}
]
```

---

## Visualization

To draw detections on the image:

```python
from utils_draw import draw_results

img = draw_results("test.jpg", results, map_class_names, output_path="test_out.jpg")
```

Output example:

<p align="center">
  <img src="data/test_out.jpg" alt="Detection Output" width="500">
</p>

---

## Repository Structure

```
.
├── export_yolo_to_onnx.py   # Script to export .pt to .onnx
├── yolo_onnx_infer.py       # YOLOONNXPredictor class for inference
├── ONNX.ipynb               # Jupyter Notebook with structured code and usage
├── data/
│   ├── test.jpg
│   └── test_out.jpg
└── README.md
```

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## Acknowledgment

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)  
- [OpenCV](https://opencv.org/)  
