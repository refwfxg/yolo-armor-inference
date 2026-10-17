#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装甲板检测与关键点推理脚本（YOLO-Pose）

功能概述
--------
逐帧读取一段装甲板视频，用 YOLO-Pose 模型对每一帧做推理，并在画面上绘制：
    1. 检测框（绿色矩形）
    2. 类别名称 + 置信度（例如 "B3 0.86"，带背景便于辨认）
    3. 4 个关键点（红色圆点）并连成四边形（黄色），即装甲板四角

最后输出一段带标注的结果视频，并在终端打印本次推理的统计信息
（帧数、有检测的帧占比、检测总数、各类别数量、平均置信度、耗时、平均速度）。

基本用法
--------
    python inference.py
    python inference.py --model best.pt --source input.mp4 --output output.mp4
    python inference.py --source my_video.mp4 --output results/my_output.mp4 --conf 0.4

参数说明
--------
    --model    模型权重路径，默认 best.pt（放在项目根目录）
    --source   输入视频路径，默认 input.mp4
    --output   结果视频保存路径，默认 output.mp4
    --conf     置信度阈值，默认 0.25
    --device   推理设备，如 cpu / 0；留空则自动选择

说明
----
运行期输出刻意保持纯 ASCII：Windows 控制台默认代码页为 GBK，直接打印中文
容易显示为乱码。中文说明集中在本文件注释与 README.md 中。
"""

import argparse
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

# 绘制颜色 (B, G, R)
COLOR_BOX = (0, 255, 0)        # 检测框：绿色
COLOR_TEXT = (255, 255, 255)   # 标签文字：白色
COLOR_TEXT_BG = (0, 140, 0)    # 标签背景：深绿
COLOR_KPT = (0, 0, 255)        # 关键点：红色
COLOR_POLY = (0, 255, 255)     # 关键点连线：黄色

KPT_RADIUS = 4
BOX_THICKNESS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Armor plate detection + 4-keypoint inference on a video (YOLO-Pose)"
    )
    parser.add_argument('--model', type=str, default='best.pt',
                        help='path to model weights (default: best.pt)')
    parser.add_argument('--source', type=str, default='input.mp4',
                        help='path to the input video (default: input.mp4)')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='path to save the annotated video (default: output.mp4)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, default='',
                        help='inference device, e.g. cpu or 0; empty means auto')
    return parser.parse_args()


def draw_label(frame, text, x, y):
    """在检测框左上角绘制带背景的标签，保证在明亮或复杂背景下也能看清。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)

    # 默认画在框的上方；上方空间不足时改画在框内顶部
    top = y - th - base - 4
    if top < 0:
        top = y + 2

    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + base + 2), COLOR_TEXT_BG, -1)
    cv2.putText(frame, text, (x + 4, top + th + 1), font, scale, COLOR_TEXT, thickness, cv2.LINE_AA)


def draw_keypoints(frame, kpts_xy):
    """绘制单个目标的关键点：先连成四边形，再画点，避免连线盖住圆点。"""
    pts = np.array([(int(p[0]), int(p[1])) for p in kpts_xy], dtype=np.int32)
    if len(pts) >= 3:
        cv2.polylines(frame, [pts], isClosed=True, color=COLOR_POLY, thickness=2)
    for (px, py) in pts:
        cv2.circle(frame, (int(px), int(py)), KPT_RADIUS, COLOR_KPT, -1)


def main():
    args = parse_args()

    # ---------- 输入校验 ----------
    if not os.path.isfile(args.model):
        print("[ERROR] Model weights not found: %s" % args.model)
        print("        Put best.pt in the project root, or pass --model <path>.")
        return 1
    if not os.path.isfile(args.source):
        print("[ERROR] Input video not found: %s" % args.source)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ---------- 加载模型 ----------
    print("[1/3] Loading model: %s" % args.model)
    model = YOLO(args.model)
    task = getattr(model, 'task', 'unknown')
    names = getattr(model, 'names', {}) or {}
    inner = getattr(model, 'model', None)
    kpt_shape = getattr(inner, 'kpt_shape', None)
    print("      task=%s  classes=%d  kpt_shape=%s" % (task, len(names), kpt_shape))
    if task != 'pose':
        print("[WARN] Model task is not 'pose'; keypoints will not be drawn.")

    predict_kwargs = {'conf': args.conf, 'verbose': False}
    if args.device:
        predict_kwargs['device'] = args.device

    # ---------- 打开输入视频 ----------
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video: %s" % args.source)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("[2/3] Input: %dx%d @ %.1f fps, %d frames (%.1f s)"
          % (width, height, fps, total_frames, (total_frames / fps) if fps else 0.0))

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        print("[ERROR] Cannot create output video: %s" % args.output)
        cap.release()
        return 1

    # ---------- 逐帧推理 ----------
    print("[3/3] Running inference (device=%s, conf=%.2f) ..."
          % (args.device or 'auto', args.conf))

    frame_idx = 0
    frames_with_det = 0
    class_counter = Counter()
    conf_list = []
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        result = model.predict(frame, **predict_kwargs)[0]

        # 1) 检测框 + 类别 + 置信度
        boxes = getattr(result, 'boxes', None)
        if boxes is not None and len(boxes) > 0:
            frames_with_det += 1
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                cls_id = int(clss[i])
                conf = float(confs[i])
                name = names.get(cls_id, str(cls_id))

                class_counter[name] += 1
                conf_list.append(conf)

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, BOX_THICKNESS)
                draw_label(frame, "%s %.2f" % (name, conf), max(0, x1), max(0, y1))

        # 2) 4 个关键点 + 四边形连线
        kpts = getattr(result, 'keypoints', None)
        if kpts is not None and getattr(kpts, 'xy', None) is not None and len(kpts.xy) > 0:
            for pts in kpts.xy.cpu().numpy():
                draw_keypoints(frame, pts)

        writer.write(frame)

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print("      processed %d/%d frames" % (frame_idx, total_frames))

    t_end = time.time()
    cap.release()
    writer.release()

    # ---------- 统计信息 ----------
    elapsed = t_end - t_start
    avg_fps = (frame_idx / elapsed) if elapsed > 0 else 0.0
    avg_conf = (sum(conf_list) / len(conf_list)) if conf_list else 0.0

    print("")
    print("=" * 62)
    print(" Inference summary")
    print("=" * 62)
    print(" model            : %s (%s)" % (args.model, task))
    print(" input video      : %s" % args.source)
    print(" output video     : %s" % args.output)
    print(" frames processed : %d / %d" % (frame_idx, total_frames))
    print(" frames with det  : %d (%.1f%%)"
          % (frames_with_det, (100.0 * frames_with_det / frame_idx) if frame_idx else 0.0))
    print(" total detections : %d" % len(conf_list))
    print(" mean confidence  : %.3f" % avg_conf)
    print(" detections/class : %s" % (dict(class_counter) if class_counter else '{}'))
    print(" elapsed          : %.1f s" % elapsed)
    print(" average speed    : %.2f fps" % avg_fps)
    print("=" * 62)
    print(" Done. Result saved to: %s" % os.path.abspath(args.output))
    return 0


if __name__ == '__main__':
    sys.exit(main())
