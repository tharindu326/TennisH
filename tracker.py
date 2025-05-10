import time
import logging
from collections import deque
import cv2
import numpy as np
from config import cfg, DetectorConfig, DetectorType

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_tracker(cfg):
    tracker_type = cfg.tracker.type.lower()
    if tracker_type == 'bytetrack':
        from trackers.bytetrack.byte_tracker import BYTETracker
        return BYTETracker(
            track_thresh=cfg.bytetrack.track_thresh,
            match_thresh=cfg.bytetrack.match_thresh,
            track_buffer=cfg.bytetrack.track_buffer,
            frame_rate=cfg.bytetrack.frame_rate
        )
    elif tracker_type == 'strongsort':
        from trackers.strongsort.strong_sort import StrongSORT
        return StrongSORT(
            model_weights=cfg.tracker.reid_weights,
            device=cfg.general.device,
            fp16=False,
            max_dist=cfg.strongsort.max_dist,
            max_iou_dist=cfg.strongsort.max_iou_dist,
            max_age=cfg.strongsort.max_age,
            max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
            n_init=cfg.strongsort.n_init,
            nn_budget=cfg.strongsort.nn_budget,
            mc_lambda=cfg.strongsort.mc_lambda,
            ema_alpha=cfg.strongsort.ema_alpha,
        )
    elif tracker_type == 'ocsort':
        from trackers.ocsort.ocsort import OCSort
        return OCSort(
            det_thresh=cfg.ocsort.det_thresh,
            max_age=cfg.ocsort.max_age,
            min_hits=cfg.ocsort.min_hits,
            iou_threshold=cfg.ocsort.iou_thresh,
            delta_t=cfg.ocsort.delta_t,
            asso_func=cfg.ocsort.asso_func,
            inertia=cfg.ocsort.inertia,
            use_byte=cfg.ocsort.use_byte,
        )
    elif tracker_type == 'boosttrack':
        from trackers.boosttrack.boost_track import BoostTrack
        return BoostTrack(
            det_thresh=cfg.boosttrack.det_thresh,
            lambda_iou=cfg.boosttrack.lambda_iou,
            lambda_mhd=cfg.boosttrack.lambda_mhd,
            lambda_shape=cfg.boosttrack.lambda_shape,
            dlo_boost_coef=cfg.boosttrack.dlo_boost_coef,
            use_dlo_boost=cfg.boosttrack.use_dlo_boost,
            use_duo_boost=cfg.boosttrack.use_duo_boost,
            max_age=cfg.boosttrack.max_age
        )
    else:
        raise ValueError(f"Undefined Tracker. Supported types: bytetrack, strongsort, ocsort, boosttrack")


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frame_id = 0
        self.avg_fps = deque(maxlen=100)
        self.tracker = initialize_tracker(cfg)

    def process_detections(self, detections: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            return np.empty((0, 6))
        mask = np.isin(detections[:, 5].astype(int), self.cfg.tracker.classes)
        return detections[mask]

    def track(self, frame: np.ndarray, detections: np.ndarray):
        filtered_detections = self.process_detections(detections)
        return self.tracker.update(filtered_detections, frame)

    def process_frame(self, frame: np.ndarray, results):
        detections = results[0].boxes.data.cpu().numpy()
        start_time = time.monotonic()
        targets = self.track(frame, detections)
        elapsed = time.monotonic() - start_time
        if elapsed > 0:
            self.avg_fps.append(1 / elapsed)
        return frame, targets


def main():
    from Yolo.detect import Detector

    video_path = 'samples/BV_S2_2201-2225.mp4'
    output_path = 'BV_S2_2201-2225_tracked.mp4'

    cfg.general.device = 'cpu'
    detector_cfg = DetectorConfig(cfg)
    player_config = detector_cfg.get(DetectorType.PLAYER)
    detector = Detector(player_config)
    tracker = Tracker(cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)
        processed_frame, tracked_objects = tracker.process_frame(frame, results)

        if cfg.flags.overlay_track:
            for obj in tracked_objects:
                x1, y1, x2, y2 = map(int, obj.tlbr)
                track_id = int(obj.track_id)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), cfg.general.COLORS['blue'], 2)
                cv2.putText(processed_frame, f'ID:{track_id}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.general.COLORS['blue'], 2)

        cv2.putText(processed_frame, f'Frame: {frame_count}', (processed_frame.shape[1] - 280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.general.TextSize, [255, 0, 0], 2, cv2.LINE_AA)

        writer.write(processed_frame)
        logger.info("Processed frame %d", frame_count)
        frame_count += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    avg_fps = np.mean(tracker.avg_fps) if tracker.avg_fps else 0
    logger.info("Tracking complete. Processed %d frames. Avg FPS: %.2f", frame_count, avg_fps)


if __name__ == '__main__':
    main()
