# -*- coding: utf-8 -*-
"""2D 统一评估入口: YOLO11 -> 统一评估器 (eval2d) 的 mAP
用法: python tools/eval_unified.py <weights> [viz_n]
- 预测用 conf=0.001/max_det=300 构建完整 PR 曲线 (AP 标准协议), NMS IoU=0.7
"""
import os
import sys
import json
import argparse
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from eval2d import load_gt_labels, evaluate_full, get_val_images
from nuscenes_common import CLASS_NAMES

DATA = os.path.join(BASE, "nu2d")


def derive_tag(weights):
    """由权重路径推出输出目录名: outputs/<tag>/train/weights/best.pt -> <tag>"""
    w = os.path.abspath(weights)
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(w))))


def predict_yolo(model, names, conf, iou, max_det, agnostic=False):
    preds = {}
    for n in names:
        r = model.predict(os.path.join(DATA, "images", "val", n), conf=conf, iou=iou,
                          max_det=max_det, agnostic_nms=agnostic, imgsz=640,
                          verbose=False)[0]
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append((int(b.cls[0].item()), x1, y1, x2, y2, float(b.conf[0].item())))
        preds[n] = boxes
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="YOLO 权重 (.pt)")
    ap.add_argument("viz_n", nargs="?", type=int, default=6)
    args = ap.parse_args()

    weights, viz_n = args.weights, args.viz_n
    tag = derive_tag(weights)

    names = get_val_images(DATA)
    sizes = {}
    for n in names:
        with Image.open(os.path.join(DATA, "images", "val", n)) as im:
            sizes[n] = im.size
    gts = load_gt_labels(os.path.join(DATA, "labels", "val"), names, sizes)

    from ultralytics import YOLO
    model = YOLO(weights)
    preds = predict_yolo(model, names, conf=0.001, iou=0.7, max_det=300)

    res = evaluate_full(preds, gts)
    print(f"== {tag} 统一评估 ({len(names)} 张验证图, AP 标准协议) ==")
    print(f"mAP@0.5      = {res['AP50']['mAP']:.4f}")
    print(f"mAP@0.5:0.95 = {res['AP5095']['mAP']:.4f}")
    print("AP50 分类别 (只列验证集有 GT 的类):")
    for k, v in res["AP50"]["per_class_ap"].items():
        gt_n = res["AP50"]["num_gt"][CLASS_NAMES.index(k)]
        if gt_n > 0:
            print(f"  {k:22s} AP={v if v is not None else '—'}  GT框数={gt_n}")

    out_json = os.path.join(BASE, "outputs", tag, "unified_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"结果已存: {out_json}")

    if viz_n > 0:
        out_dir = os.path.join(BASE, "outputs", "viz_check")
        os.makedirs(out_dir, exist_ok=True)
        dense = sorted(names, key=lambda n: -len(gts[n]))
        for n in dense[:viz_n]:
            # 可视化用更紧阈值 + 跨类别抑制, 消除同目标重复框; 正式评估协议不变
            vp = predict_yolo(model, [n], conf=0.10, iou=0.45, max_det=100,
                              agnostic=True)[n]
            im = Image.open(os.path.join(DATA, "images", "val", n)).convert("RGB")
            d = ImageDraw.Draw(im)
            for c, x1, y1, x2, y2 in gts[n]:
                d.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
            for c, x1, y1, x2, y2, s in vp:
                d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=3)
                d.text((x1, max(0, y1 - 12)), f"{s:.2f}", fill=(255, 40, 40))
            im.save(os.path.join(out_dir, n))
        print(f"可视化 {viz_n} 张 (GT 最多的图) -> outputs/viz_check/ "
              f"绿=GT 红=预测, 更紧阈值 + 跨类抑制")


if __name__ == "__main__":
    main()
