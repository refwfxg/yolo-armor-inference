import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv26 Armor Inference")
    parser.add_argument('--model', type=str, default='best.pt', help='Path to model weights')
    parser.add_argument('--source', type=str, default='input.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save output video')
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Error: Cannot open video. Check input path and filename.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("Starting inference...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        results = model(frame, verbose=False)
        result = results[0]

        # 画框和标签
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 画4个关键点（重点）
        if result.keypoints is not None:
            for kpt in result.keypoints.xy[0]:
                cx, cy = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        out.write(frame)
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames}...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Result saved to {args.output}")


if __name__ == '__main__':
    main()