from pathlib import Path
from typing import List, Dict, Union, Optional
import cv2
import numpy as np
import onnxruntime as ort

class YOLOONNXPredictor:
    def __init__(self, onnx_path: Union[str, Path], providers: Optional[List[str]] = None, imgsz: int = 640, class_names: Optional[Dict[int, str]] = None):
        self.onnx_path = str(onnx_path)
        self.session = ort.InferenceSession(self.onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shp = self.session.get_inputs()[0].shape
        self.H = int(shp[2]) if isinstance(shp[2], int) else imgsz
        self.W = int(shp[3]) if isinstance(shp[3], int) else imgsz
        self.class_names = class_names

    def _letterbox(self, img, new_shape):
        h0, w0 = img.shape[:2]
        r = min(new_shape[0] / h0, new_shape[1] / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        imr = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (new_shape[0] - nh) // 2
        bottom = new_shape[0] - nh - top
        left = (new_shape[1] - nw) // 2
        right = new_shape[1] - nw - left
        imp = cv2.copyMakeBorder(imr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return imp, r, (left, top), (h0, w0)

    def _nms(self, boxes, scores, iou_thres):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return keep

    def _normalize_output(self, out):
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        if out.ndim == 2 and out.shape[0] in (6, 7, 8) and out.shape[1] > out.shape[0]:
            out = out.T
        if out.ndim == 2 and out.shape[1] > out.shape[0] and out.shape[1] <= 1024 and out.shape[0] <= 100000:
            pass
        elif out.ndim == 2 and out.shape[0] > out.shape[1]:
            pass
        elif out.ndim == 2 and out.shape[0] in (84, 85) and out.shape[1] > 1000:
            out = out.T
        return out.astype(np.float32, copy=False)

    def _postprocess(self, out, conf_thres, iou_thres, hw0, ratio, dwdh):
        out = self._normalize_output(out)
        if out.ndim != 2 or out.shape[1] < 6:
            return []
        C = out.shape[1]
        xywh = out[:, :4]
        if self.class_names is not None:
            k = len(self.class_names)
            has_obj = (C == 5 + k)
            no_obj = (C == 4 + k)
        else:
            k = C - 5 if C >= 6 else C - 4
            has_obj = C == 5 + k and k > 0
            no_obj = C == 4 + k and k > 0
        if has_obj:
            obj = np.clip(out[:, 4:5], 0.0, 1.0)
            cls_probs = out[:, 5:]
            cls_ids = np.argmax(cls_probs, axis=1)
            cls_scores = np.take_along_axis(cls_probs, cls_ids[:, None], axis=1)[:, 0]
            scores = np.clip(obj[:, 0] * cls_scores, 0.0, 1.0)
        elif no_obj:
            cls_probs = out[:, 4:]
            cls_ids = np.argmax(cls_probs, axis=1)
            cls_scores = np.take_along_axis(cls_probs, cls_ids[:, None], axis=1)[:, 0]
            scores = np.clip(cls_scores, 0.0, 1.0)
        else:
            if C >= 6:
                obj = np.clip(out[:, 4:5], 0.0, 1.0)
                cls_probs = out[:, 5:]
                cls_ids = np.argmax(cls_probs, axis=1)
                cls_scores = np.take_along_axis(cls_probs, cls_ids[:, None], axis=1)[:, 0]
                scores = np.clip(obj[:, 0] * cls_scores, 0.0, 1.0)
            else:
                cls_probs = out[:, 4:]
                cls_ids = np.argmax(cls_probs, axis=1)
                cls_scores = np.take_along_axis(cls_probs, cls_ids[:, None], axis=1)[:, 0]
                scores = np.clip(cls_scores, 0.0, 1.0)
        m = scores >= conf_thres
        if not np.any(m):
            return []
        xywh = xywh[m]
        scores = scores[m]
        cls_ids = cls_ids[m]
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        dw, dh = dwdh
        xyxy[:, [0, 2]] -= dw
        xyxy[:, [1, 3]] -= dh
        xyxy /= ratio
        h0, w0 = hw0
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w0 - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h0 - 1)
        keep = self._nms(xyxy, scores, iou_thres)
        xyxy = xyxy[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        results = []
        for (x1, y1, x2, y2), sc, cid in zip(xyxy, scores, cls_ids):
            xc = ((x1 + x2) / 2) / w0
            yc = ((y1 + y2) / 2) / h0
            w = (x2 - x1) / w0
            h = (y2 - y1) / h0
            item = {"x": float(xc), "y": float(yc), "w": float(w), "h": float(h), "conf": float(sc), "cls": int(cid)}
            if self.class_names is not None:
                item["label"] = self.class_names.get(int(cid), str(int(cid)))
            results.append(item)
        return results

    def predict(self, image: Union[str, Path, np.ndarray], conf_thres: float = 0.25, iou_thres: float = 0.5) -> List[Dict]:
        img0 = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
        if img0 is None:
            return []
        im_padded, ratio, dwdh, hw0 = self._letterbox(img0, (self.H, self.W))
        im = cv2.cvtColor(im_padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        im = np.expand_dims(im, 0)
        out = self.session.run(None, {self.input_name: im})[0]
        return self._postprocess(out, conf_thres, iou_thres, hw0, ratio, dwdh)
