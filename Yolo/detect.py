from ultralytics import YOLO
import cv2
import time
import numpy as np
import os
from pathlib import Path
import sys
main_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(main_dir))
from config import cfg
import torch
from thop import profile

'''
USAGE
python Yolo/detect.py -i samples/frame_1.png -o results/VR-samples/frame_1.png
'''

def draw_boxes(img, boxes, labels, confidences, class_ids, color=(0, 255, 0), thickness=2):
    draw_img = img.copy()
    for box, label, conf, class_id in zip(boxes, labels, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = cfg.general.COLORS[list(cfg.general.COLORS)[int(class_id) % len(cfg.general.COLORS)]]
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        text = f"{label}: {conf:.2f}"
        cv2.putText(draw_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, color, 1, cv2.LINE_AA)
    return draw_img


class Detector:
    def __init__(self):
        self.model = YOLO(os.path.join(cfg.detection_player.model))
        
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.detect(img)

    def detect(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        results = self.model.predict(source=img, conf=cfg.detection_player.OBJECTNESS_CONFIDANCE,
                                     iou=cfg.detection_player.NMS_THRESHOLD,
                                     classes=cfg.detection_player.classes,
                                     device=cfg.general.device,
                                     verbose=cfg.detection_player.verbose, 
                                     max_det=cfg.detection_player.max_det)
        
        detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
        boxes = detections[:, :-2].astype(int)
        # print(f"Detection inference took: {time.perf_counter() - start:.4f} seconds")
                
        labels = [self.model.names[int(cls)] for cls in results[0].boxes.cls]
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        if cfg.flags.render_detections:
            img = draw_boxes(img, boxes, labels, confidences, class_ids)
            
        return results, img


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Object detection with YOLO")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to an image file or a folder of images to process")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the processed output images")
    args = parser.parse_args()

    detector = Detector()

    if os.path.isfile(args.input):
        # Process a single image
        img = cv2.imread(args.input)
        if img is None:
            print(f"Error: Unable to load image {args.input}")
        else:
            results = detector.detect(img)
            detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
            boxes = detections[:, :-2].astype(int)
            class_ids = detections[:, 5].astype(int)
            
            labels = [detector.model.names[int(cls)] for cls in results[0].boxes.cls]
            confidences = results[0].boxes.conf.cpu().numpy()
            output_img = draw_boxes(img, boxes, labels, confidences, class_ids)
            output_path = os.path.join(args.output, os.path.basename(args.input))
            cv2.imwrite(output_path, output_img)
            print(f"Processed {args.input}, saved to {output_path}")
    elif os.path.isdir(args.input):
        # Process all images in the folder
        os.makedirs(args.output, exist_ok=True)
        for file_name in os.listdir(args.input):
            file_path = os.path.join(args.input, file_name)
            img = cv2.imread(file_path)
            if img is None:
                print(f"Skipping {file_name}: Unable to load image")
                continue
            results = detector.detect(img)
            detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
            boxes = detections[:, :-2].astype(int)
            
            labels = [detector.model.names[int(cls)] for cls in results[0].boxes.cls]
            confidences = results[0].boxes.conf.cpu().numpy()
            output_img = draw_boxes(img, boxes, labels, confidences)
            output_path = os.path.join(args.output, file_name)
            cv2.imwrite(output_path, output_img)
            print(f"Processed {file_name}, saved to {output_path}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")

    