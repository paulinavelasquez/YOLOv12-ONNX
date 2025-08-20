#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO .pt to ONNX (+ quick onnxruntime test)")
    p.add_argument("--weights", type=str, required=True, help="Path to .pt weights (Ultralytics)")
    p.add_argument("--imgsz", type=int, nargs=2, default=[640, 640], help="Input size H W (default 640 640)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset version (11/12/13/17)")
    p.add_argument("--dynamic", action="store_true", help="Dynamic axes (batch/height/width)")
    p.add_argument("--half", action="store_true", help="Export in FP16 if possible")
    p.add_argument("--simplify", action="store_true", help="Simplify ONNX with onnxsim")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    p.add_argument("--outfile", type=str, default="", help="Optional output .onnx path")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception:
        print("ERROR: Could not import ultralytics. Install with: pip install ultralytics", file=sys.stderr)
        raise

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    model = YOLO(str(weights))

    if args.outfile:
        onnx_path = Path(args.outfile)
    else:
        onnx_path = weights.with_suffix(".onnx")

    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "opset": args.opset,
        "dynamic": args.dynamic,
        "half": args.half,
        "device": args.device,
    }

    print("=== Exporting to ONNX ===")
    print(f"Weights:     {weights}")
    print(f"Output ONNX: {onnx_path}")
    print(f"Args:        imgsz={args.imgsz}, opset={args.opset}, dynamic={args.dynamic}, half={args.half}, device={args.device}")

    exported = model.export(**export_kwargs)

    cand = None
    if isinstance(exported, (list, tuple)):
        for x in exported:
            if str(x).endswith(".onnx"):
                cand = Path(x)
                break
    elif isinstance(exported, (str, Path)):
        pth = Path(exported)
        cand = pth if pth.suffix == ".onnx" else None

    if cand is None:
        cands = sorted(Path(".").rglob("*.onnx"), key=os.path.getmtime, reverse=True)
        if cands:
            cand = cands[0]
    if cand is None or not cand.exists():
        raise RuntimeError("ONNX file not found. Check Ultralytics export output.")

    if onnx_path.resolve() != cand.resolve():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(cand), str(onnx_path))

    print(f"ONNX file saved at: {onnx_path.resolve()}")

    if args.simplify:
        try:
            print("=== Simplifying ONNX with onnxsim ===")
            import onnx
            from onnxsim import simplify
            model_onnx = onnx.load(str(onnx_path))
            model_simp, check = simplify(model_onnx)
            if not check:
                print("Warning: onnxsim check failed, keeping original model.")
            else:
                onnx.save(model_simp, str(onnx_path))
                print("Simplification completed.")
        except Exception as e:
            print(f"Warning: could not simplify with onnxsim: {e}")

    try:
        print("=== Testing with onnxruntime (dummy inference) ===")
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        h, w = args.imgsz
        dummy = (np.random.rand(1, 3, h, w).astype("float16" if args.half else "float32"))
        outs = sess.run(None, {input_name: dummy})
        print(f"Inference OK. Number of outputs: {len(outs)}")
        for i, o in enumerate(outs):
            print(f"  - output[{i}] -> shape={o.shape}, dtype={o.dtype}")
        print("Quick validation completed.")
    except Exception as e:
        print(f"Warning: onnxruntime not available or test failed: {e}")

if __name__ == "__main__":
    main()
