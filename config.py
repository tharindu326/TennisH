#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict
from enum import Enum

__C = edict()
cfg = __C


# detector inference
__C.detection_player = edict()
__C.detection_player.model = 'model_data/best_player.pt'
__C.detection_player.classes = [0, 1, 2, 3]  # filter by class: --class 0, or --class 0 2 3
__C.detection_player.OBJECTNESS_CONFIDANCE = 0.2
__C.detection_player.NMS_THRESHOLD = 0.45
__C.detection_player.verbose = False
__C.detection_player.max_det = 10

__C.detection_scoreboard = edict()
__C.detection_scoreboard.model = 'model_data/best_score_box.pt'
__C.detection_scoreboard.classes = [0]  # filter by class: --class 0, or --class 0 2 3
__C.detection_scoreboard.OBJECTNESS_CONFIDANCE = 0.2
__C.detection_scoreboard.NMS_THRESHOLD = 0.45
__C.detection_scoreboard.verbose = False
__C.detection_scoreboard.max_det = 1

class DetectorType(Enum):
    PLAYER = "detection_player"
    SCOREBOARD = "detection_scoreboard"

class DetectorConfig:
    def __init__(self, cfg_dict):
        self.cfg = cfg_dict

    def get(self, detector_type: DetectorType):
        return getattr(self.cfg, detector_type.value)
    
__C.general = edict()
__C.general.TextSize = 1  # front size for frame number and INs/ OUTs overlay
__C.general.frame_rotate = False
__C.general.frame_resize = None  # (640, 640)
__C.general.COLORS = {
                          'green': [64, 255, 64],
                          'blue': [255, 128, 0],
                          'coral': [0, 128, 255],
                          'yellow': [0, 255, 255],
                          'gray': [169, 169, 169],
                          'cyan': [255, 255, 0],
                          'magenta': [255, 0, 255],
                          'white': [255, 255, 255],
                          'red': [64, 0, 255]
                      }
__C.general.model_path = 'model_data'
__C.general.device = 'cpu'  # if GPU give the device ID; EX: , else 'cpu'
__C.general.output_path = 'outputs'

# overlay Flags
__C.flags = edict()
__C.flags.image_show = True
__C.flags.render_detections = True
__C.flags.render_fps = False
__C.flags.overlay_track = True
__C.flags.overlay_highlights = True
__C.flags.overlay_score = True

# video inference option
__C.video = edict()
__C.video.video_writer_fps = 30
__C.video.FOURCC = 'mp4v'  # 'avc1'  # 4-byte code used to specify the video codec
__C.video.requiredFPS = 20
__C.video.save = True
__C.video.FPS = 30  # FPS of the source

# Trackers
__C.tracker = edict()
__C.tracker.type = 'bytetrack'  # 'boosttrack', 'bytetrack'  'ocsort' Select the tracker. But here we only use bytetrack
__C.tracker.classes = [3]  # classes id to track
__C.tracker.reid_weights = 'model_data/osnet_x0_25_msmt17.pt'
__C.tracker.time_since_update_threshold = 6  # last update of the track is before 6 frames it will still consider unconfirmed tracks as active
__C.tracker.trail_length = 60
__C.tracker.enable = True
__C.tracker.student_reinit_iou_threshold = 0.1

# ByteTracker: In use
__C.bytetrack = edict()
__C.bytetrack.track_thresh = 0.2  # if the confidence_score> track_thresh + det_tresh_gap then initialize a new track otherwise it will only match tracklets where confidance_score> track_thresh
__C.bytetrack.track_buffer = 30  # length of maximum frames where can a lost tracklet be. if the tracklet not appear within 30 frames track will be deleted., else track will rebirth
__C.bytetrack.match_thresh = 0.95  # linear assignment threshold where it uses Jonker-Volgenant algorithm. when this is lower no tracklets age. this can be also defined as cost of assignment of the Jonker-Volgenant algorithm. maximum error that allow for the linear assignment.
__C.bytetrack.frame_rate = 30  # frame rate of the video; used to define the buffer size; buffer_size = int(frame_rate / 30.0 * self.track_buffer)


# OCSORT and Strong sort are not in use. They are just optional trackers, we are only using ByteTrack
# StrongSort
__C.strongsort = edict()
__C.strongsort.ecc = True
__C.strongsort.ema_alpha = 0.8962157769329083
__C.strongsort.max_age = 40
__C.strongsort.max_dist = 0.1594374041012136
__C.strongsort.max_iou_dist = 0.5431835667667874
__C.strongsort.max_unmatched_preds = 0
__C.strongsort.mc_lambda = 0.995
__C.strongsort.n_init = 3
__C.strongsort.nn_budget = 100
__C.strongsort.conf_thres = 0.5122620708221085

# OCSort
__C.ocsort = edict()
__C.ocsort.asso_func = 'giou'
__C.ocsort.conf_thres = 0.5122620708221085
__C.ocsort.delta_t = 1
__C.ocsort.det_thresh = 0
__C.ocsort.inertia = 0.3941737016672115
__C.ocsort.iou_thresh = 0.22136877277096445
__C.ocsort.max_age = 50
__C.ocsort.min_hits = 1
__C.ocsort.use_byte = False

# BoostTrack

__C.boosttrack = edict()
__C.boosttrack.det_thresh = 0.2
__C.boosttrack.lambda_iou = 0.2
__C.boosttrack.lambda_mhd = 0.25
__C.boosttrack.lambda_shape = 0.25
__C.boosttrack.dlo_boost_coef = 0.65
__C.boosttrack.use_dlo_boost = True
__C.boosttrack.use_duo_boost = True
__C.boosttrack.max_age = 30

__C.OCR = edict()
__C.OCR.preprocess = True

__C.highlights = edict()
__C.highlights.debounce_secs_for_score_change = 1
__C.highlights.easy_AD = True
__C.highlights.easy_AD_percentage = 0.2
