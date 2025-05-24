import os 
import sys
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import logging
logging.getLogger('ppocr').setLevel(logging.ERROR)


class PaddleOCRProcessor:
    def __init__(self, lang='en', use_angle_cls=False, use_gpu=False, gpu_mem=6000):
        self.ocr = PaddleOCR(
                                use_angle_cls=use_angle_cls,
                                use_gpu=use_gpu,
                                gpu_mem=gpu_mem,
                                lang=lang
                            )

    def process_image(self, image):
        ret = {"boxes": [], "texts": [], "scores": []}
        score = None

        result = self.ocr.ocr(image, cls=True)
        if not (result and result[0]):
            return ret, score

        # unpack OCR
        boxes = [line[0] for line in result[0]]
        texts = [line[1][0].strip() for line in result[0]]
        confs = [line[1][1]      for line in result[0]]
        ret = {"boxes": boxes, "texts": texts, "scores": confs}
        
        # pull names (first two non-numeric, non-adv tokens)
        names = []
        for t in texts:
            if re.fullmatch(r"\d+", t): continue
            if re.fullmatch(r"(?i)(AD|A|ADV|ADVANTAGE)", t): continue
            names.append(t)
            if len(names) == 2:
                break

        # build (text, x_center, y_center) list
        toks = []
        for box, text in zip(boxes, texts):
            if not text: continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            toks.append((text, np.mean(xs), np.mean(ys)))

        if len(toks) < 2:
            return ret, score

        # split into two rows by median y
        ys    = [y for _, _, y in toks]
        med_y = np.median(ys)
        row1  = [t for t in toks if t[2] <  med_y]
        row2  = [t for t in toks if t[2] >= med_y]

        # extract each row's numeric/ADV tokens *with* their x-center
        def extract_numeric(row):
            out = []
            for txt, x, _ in row:
                if re.fullmatch(r"\d+", txt):
                    out.append((int(txt), x))
                elif re.fullmatch(r"(?i)(AD|A|ADV|ADVANTAGE)", txt):
                    out.append((50, x))
            return out

        num_x1 = extract_numeric(row1)
        num_x2 = extract_numeric(row2)

        # align the shorter row to the longer one, filling blanks with 0
        def align_numeric_rows(rx1, rx2):
            # choose reference = the row with more tokens
            if len(rx1) >= len(rx2):
                ref, other, ref_is1 = sorted(rx1, key=lambda v_x: v_x[1]), sorted(rx2, key=lambda v_x: v_x[1]), True
            else:
                ref, other, ref_is1 = sorted(rx2, key=lambda v_x: v_x[1]), sorted(rx1, key=lambda v_x: v_x[1]), False

            k = len(ref)
            # trivial: 0 or 1 column → just sort
            if k <= 1:
                sp_ref   = [v for v, _ in ref]
                sp_other = [v for v, _ in other]
                return (sp_ref, sp_other) if ref_is1 else (sp_other, sp_ref)

            # build a matching threshold = half the smallest gap between ref-columns
            xs   = [x for _, x in ref]
            diffs = [xs[i+1] - xs[i] for i in range(k-1)]
            thresh = min(diffs) / 2

            # assign each ref-col either a real value or 0
            used = [False]*len(other)
            sp_other = []
            for rx in xs:
                best_i, best_d = None, float('inf')
                for i, (_, ox) in enumerate(other):
                    if used[i]: continue
                    d = abs(ox - rx)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None and best_d <= thresh:
                    sp_other.append(other[best_i][0])
                    used[best_i] = True
                else:
                    sp_other.append(0)

            sp_ref = [v for v, _ in ref]
            return (sp_ref, sp_other) if ref_is1 else (sp_other, sp_ref)

        nums1, nums2 = align_numeric_rows(num_x1, num_x2)

        # figure out set vs point for each (unchanged)
        def pick(sp):
            if   len(sp) == 0:     return 0, 0
            elif len(sp) == 1:     return 0, sp[0]
            elif len(sp) == 2:     return sp[0], sp[1]
            # len>=3: [old_set, curr_set, point, ...]
            return sp[1], sp[2]

        s1, p1 = pick(nums1)
        s2, p2 = pick(nums2)
        
        if p1 == 50 and p2 == 0:
            p2 = 40
        elif p2 == 50 and p1 == 0:
            p1 = 40

        score = {
            "player1":    names[0].lower() if len(names) > 0 else None,
            "player2":    names[1].lower() if len(names) > 1 else None,
            "set_score":   [s1, s2],
            "point_score": [p1, p2]
        }

        return ret, score
    
    def draw_score(self, im, ocr_results, box):
        boxes = ocr_results["boxes"]
        txts = ocr_results["texts"]
        scores = ocr_results["scores"]
        if box and len(boxes) != 0:
            # Adjust OCR boxes by adding score board box coordinates
            adjusted_boxes = []
            x_offset, y_offset = box[0], box[1]
            for ocr_box in boxes:
                adjusted_box = [[x + x_offset, y + y_offset] for x, y in ocr_box]
                adjusted_boxes.append(adjusted_box)
            boxes = adjusted_boxes
        if boxes:
            font_scale = sum(im.shape[:2]) / 2500
            line_width = max(round(sum(im.shape[:2]) / 2000), 2)
            y_offset = int(box[1]) - 20  # Starting position for the text
            for text, conf in zip(txts, scores):
                label_text = f"{text} | {conf:.3f}"
                cv2.putText(im, label_text, (int(box[0]), y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, [0, 0, 255], line_width)
                y_offset += int(font_scale * 40)
            for box_ in boxes:
                p1, p2 = tuple(map(int, box_[0])), tuple(map(int, box_[2]))
                cv2.rectangle(im, p1, p2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        return im
    
    def preprocess_image(self, image):
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Apply Gaussian blur to reduce noise (but not too much!)
        blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

        # 3. Adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return thresh


if __name__ == "__main__":
    from Yolo.detect import Detector
    from config import cfg, DetectorConfig, DetectorType
    detector_cfg = DetectorConfig(cfg)
    scoreboard_config = detector_cfg.get(DetectorType.SCOREBOARD)
    detector = Detector(scoreboard_config)
    
    processor = PaddleOCRProcessor(lang='en')
    source = 'tk1.png'
    out_dir = 'results/with_plates'
    
    # source = 'samples/LP5.jpeg'
    # out_dir = 'results/'
    
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.isfile(source):
        image = cv2.imread(source)
        file_name = os.path.basename(source)
        results, frame = detector.detect(image)
        detections = results[0].boxes.data.cpu().numpy()
        boxes = detections[:, :-2].astype(int) 
        out_path = os.path.join(out_dir, f'predicted_{file_name}')
        for i, box in enumerate(boxes):
            x, y, x2, y2 = map(int, box)
            w, h = x2 - x, y2 - y
            score_image = frame[y: y + h, x: x + w]
            ocr_results = processor.process_image(score_image)
            print(ocr_results)
    
    # elif os.path.isdir(source):
    #     for file in os.listdir(source):
    #         file_name = ".".join(file.split(".")[:-1])
    #         im_path = os.path.join(source, file)
    #         out_path = os.path.join(out_dir, f'predicted_{file_name}.jpg')
    #         image = cv2.imread(im_path)
    #         frame_out, processed_boxes, processed_confidence, processed_class_id = inference.infer(image)
    #         for lp_box in processed_boxes['LP']:
    #             print(f'Detected LP: {lp_box}')
    #             lp_x_c, lp_y_c, lp_w, lp_h = lp_box
    #             lp_x, lp_y = int(lp_x_c - lp_w / 2), int(lp_y_c - lp_h / 2)
    #             lp_image = image[lp_y: lp_y + lp_h, lp_x: lp_x + lp_w]
    #             # lp_image = processor.preprocess_lp_image(lp_image)
    #             ocr_results = processor.process_image(lp_image)
    #             # output_image = processor.draw_results(frame_out, ocr_results, LP_boxes=[lp_x, lp_y, lp_w, lp_h], font_path='OCR/simfang.ttf')
    #             output_image = processor.draw_LP_on_vehicle(frame_out, ocr_results, [lp_x, lp_y, lp_w, lp_h])
    #             cv2.imwrite(out_path, output_image)
        
        
