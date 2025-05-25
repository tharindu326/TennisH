# Tennis Analysis System

A comprehensive Python-based tennis video analysis system that provides real-time player tracking, score detection, and highlight generation using computer vision and machine learning techniques.

![system](data/front.png).

## Features

- **Real-time Player Tracking**: Multi-object tracking using ByteTrack algorithm
- **Automatic Score Detection**: OCR-based scoreboard reading with validation
- **Highlight Generation**: Automatic detection of key moments including:
  - Aces and service winners
  - Break points and advantage points
  - Extended rallies
  - Game and set endings
- **Ball Trajectory Analysis**: Track ball movement and analyze court positioning
- **CSV Export**: Detailed match statistics and event logging
- **Visual Overlays**: Real-time display of scores, player tracking, and highlights

## Installation

### Python Installation
Ensure you have Python 3.8 or higher installed on your system.

### Library Installation
Install required dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

## Running the Script

### Clone the Repository

```bash
git clone https://github.com/tharindu326/TennisH.git
cd TennisH
```

Execute the main script with your tennis video file:

```bash
python .\main.py -s test3.mp4
```

### Command Line Arguments
- `-s` or `--source`: Path to the input tennis video file

## Configuration

The system uses a comprehensive configuration system located in `config.py`. Below are the main configuration categories and their parameters:

### Detection Player Configuration
- **model**: `'model_data/best_yolov8n.pt'` - Path to YOLOv8 detection model
- **classes**: `[0, 1, 2, 3, 4]` - Object classes to detect (ball, net, player, racket, scoreboard)
- **OBJECTNESS_CONFIDANCE**: `0.2` - Minimum confidence threshold for object detection
- **NMS_THRESHOLD**: `0.45` - Non-maximum suppression threshold
- **verbose**: `False` - Enable/disable verbose logging
- **max_det**: `10` - Maximum number of detections per frame

### General Configuration
- **TextSize**: `1` - Font size for frame number and overlay text
- **frame_rotate**: `False` - Enable frame rotation
- **frame_resize**: `None` - Frame resize dimensions (e.g., (640, 640))
- **COLORS**: Dictionary of color definitions for visualization overlays
- **model_path**: `'model_data'` - Directory containing model files
- **device**: `'cpu'` - Processing device ('cpu' or GPU device ID)
- **output_path**: `'outputs'` - Directory for output files

### Overlay Flags
- **image_show**: `False` - Display processed frames in real-time
- **render_detections**: `True` - Show object detection bounding boxes
- **render_fps**: `False` - Display FPS counter
- **overlay_player_track**: `True` - Show player tracking trails
- **overlay_ball_track**: `True` - Show ball trajectory
- **overlay_highlights**: `True` - Display highlight notifications
- **overlay_score**: `True` - Show current score overlay

### Video Configuration
- **video_writer_fps**: `30` - Output video frame rate
- **FOURCC**: `'mp4v'` - Video codec for output files
- **requiredFPS**: `20` - Minimum required FPS for processing
- **save**: `True` - Enable video output saving
- **FPS**: `30` - Source video frame rate

### Tracker Configuration
- **type**: `'bytetrack'` - Tracking algorithm (supports 'bytetrack', 'boosttrack', 'ocsort')
- **classes**: `[]` - Specific classes to track (empty = all classes)
- **reid_weights**: `'model_data/osnet_x0_25_msmt17.pt'` - Re-identification model weights
- **time_since_update_threshold**: `6` - Frames before considering track lost
- **trail_length**: `60` - Length of tracking trail visualization
- **enable**: `True` - Enable/disable tracking
- **student_reinit_iou_threshold**: `0.1` - IoU threshold for track re-initialization
- **ball_track**: `True` - Enable ball tracking

### ByteTracker Configuration
- **track_thresh**: `0.2` - Minimum confidence for track initialization
- **track_buffer**: `30` - Maximum frames to keep lost tracks
- **match_thresh**: `0.95` - Linear assignment threshold for track matching
- **frame_rate**: `30` - Video frame rate for buffer calculations

### StrongSort Configuration (Optional)
- **ecc**: `True` - Enable Enhanced Correlation Coefficient
- **ema_alpha**: `0.8962157769329083` - Exponential moving average factor
- **max_age**: `40` - Maximum age for tracks
- **max_dist**: `0.1594374041012136` - Maximum distance for matching
- **max_iou_dist**: `0.5431835667667874` - Maximum IoU distance
- **max_unmatched_preds**: `0` - Maximum unmatched predictions
- **mc_lambda**: `0.995` - Motion compensation lambda
- **n_init**: `3` - Number of frames for track initialization
- **nn_budget**: `100` - Nearest neighbor budget
- **conf_thres**: `0.5122620708221085` - Confidence threshold

### OCSort Configuration (Optional)
- **asso_func**: `'giou'` - Association function (GIoU)
- **conf_thres**: `0.5122620708221085` - Confidence threshold
- **delta_t**: `1` - Time delta for prediction
- **det_thresh**: `0` - Detection threshold
- **inertia**: `0.3941737016672115` - Track inertia factor
- **iou_thresh**: `0.22136877277096445` - IoU threshold
- **max_age**: `50` - Maximum track age
- **min_hits**: `1` - Minimum hits for track confirmation
- **use_byte**: `False` - Use ByteTrack features

### BoostTrack Configuration (Optional)
- **det_thresh**: `0.2` - Detection threshold
- **lambda_iou**: `0.2` - IoU weight factor
- **lambda_mhd**: `0.25` - Mahalanobis distance weight
- **lambda_shape**: `0.25` - Shape similarity weight
- **dlo_boost_coef**: `0.65` - DLO boost coefficient
- **use_dlo_boost**: `True` - Enable DLO boost
- **use_duo_boost**: `True` - Enable DUO boost
- **max_age**: `30` - Maximum track age

### OCR Configuration
- **preprocess**: `True` - Enable image preprocessing for OCR

### Highlights Configuration
- **debounce_secs_for_score_change**: `0.2` - Debounce time for score changes
- **extended_rally_energy_threshold**: `2000` - Energy threshold for extended rally detection

### Players Configuration
- **confirm_threshold**: `10` - Frames needed to confirm new player track
- **deactivation_timeout**: `3` - Seconds without update before considering player lost

## System Outputs

The system generates several types of outputs:

### Video Output
- Processed video with overlays showing:
  - Player tracking boxes and IDs
  - Ball trajectory visualization
  - Real-time score display
  - Highlight notifications
  - Game statistics

### CSV Export
Detailed match statistics exported to CSV format including:
- Start and end times for each point
- Player movements and positions
- Score progression
- Event detection (aces, break points, etc.)
- Energy metrics for rally analysis

### Highlight Detection
Automatic identification of key moments:
- **Aces**: Service points won without opponent contact
- **Service Winners**: Points won on serve with minimal return
- **Break Points**: Critical scoring opportunities
- **Extended Rallies**: High-energy, long-duration points
- **Game Endings**: Conclusion of individual games
- **Set Endings**: Conclusion of sets

### Real-time Analytics
- Player movement energy calculation for extended rallies detection 
- Ball trajectory analysis for Aces and Service winners
- Score validation (for OCR improvements) and progression tracking

## Getting Started

For detailed setup instructions and advanced usage examples, refer to the `run_TennisH.ipynb` notebook which provides:

- Step-by-step dependency installation
- Sample video processing workflows

# Highlights Generating Strategy

The Tennis Analysis System automatically generates highlights by analyzing video data to detect key events. Each event type triggers the creation of a specific video clip segment, cropped based on the following ideal strategies:

| **Event**          | **Ideal Cropping Strategy**                           | **Padding**             |
| ------------------ | ----------------------------------------------------- | ----------------------- |
| **Ace**            | Serve → ball bounce → brief reaction                  | **1s before, 2s after** |
| **Service Winner** | Serve → quick point end                               | **1s before, 2s after** |
| **Break Point**    | Entire rally, emphasizing tension and reaction        | **2s before, 3s after** |
| **Advantage**      | Full rally (after deuce)                              | **2s before, 2s after** |
| **Game Ending**    | Entire rally + celebration (reaction/player close-up) | **3s before, 4s after** |
| **Set Ending**     | Full rally + extended celebration (crowd)             | **5s before, 5s after** |
| **Extended Rally** | Full length of unusually exciting rally               | **1s before, 1s after** |

When multiple highlight events occur simultaneously for a single score (e.g., a break point combined with an extended rally), the system merges these events into a single clip. In such cases, the longest padding required among all triggered events is used, ensuring comprehensive context and capturing maximum excitement in each highlight.

These smaller clips are individually saved to timestamped directories within the output folder as they are generated during video processing. At the end of the processing, all these clips are automatically merged into a single highlights video (`combined_highlights.mp4`)