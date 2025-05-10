import cv2
from Yolo.detect import Detector
from tracker import Tracker
from config import cfg, DetectorConfig, DetectorType
from OCR.paddle_OCR import PaddleOCRProcessor
from collections import deque
import csv
import os 
from datetime import datetime


class Highlights:
    def __init__(self, input_video_path):
        self.tracker = PlayerTracker()
        self.scorer = ScoreReader()
        self.points = Points()
        self.score_history = {}
        self.highlights = []
        self.debounce_secs = cfg.highlights.debounce_secs_for_score_change
        self._score_queue = deque()
        self._last_committed_score = None
        
        video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(cfg.general.output_path, f"{video_basename}_{ts_str}")
        os.makedirs(self.output_dir, exist_ok=True)
        csv_filename = f"{video_basename}_{ts_str}.csv"
        self.csv_path = os.path.join(self.output_dir, csv_filename)

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'highlight_type',
                'player',
                'set_score',
                'point_score'
            ])

    def player_track(self, frame):
        """Track players in the frame"""
        return self.tracker.track_players(frame)

    def get_score(self, frame, video_timestamp): 
        """
        Process the frame to extract score and detect break/end points
        """
        frame, new_score = self.scorer.get_score(frame)
        if not new_score:
            return frame
        if not self._is_valid_score(new_score):
            return frame
        
        if self._debounce_and_commit_score(new_score, video_timestamp):
            self.score_history[video_timestamp] = new_score
            highlights = self.points.process(video_timestamp, new_score)
            htypes = []
            if highlights:
                htypes = list(highlights.keys())
                self.highlights.append({
                    'timestamp': video_timestamp,
                    'types': highlights.keys(),
                    'score': new_score
                })
                
                if cfg.flags.overlay_highlights:
                    self._draw_highlight_overlay(frame, highlights)

            # LIVE‐APPEND to CSV: two rows per event
            s1, s2 = new_score['set_score']
            p1, p2 = new_score['point_score']
            p1_name = new_score.get('player1')
            p2_name = new_score.get('player2')
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                # row 1: player 1
                writer.writerow([
                    video_timestamp,
                    ','.join(htypes),  # empty string if no highlight
                    p1_name,
                    s1,
                    p1
                ])
                # row 2: player 2 (blank ts & hl-type)
                writer.writerow([
                    '',
                    '',
                    p2_name, 
                    s2,
                    p2
                ])
                    
        if cfg.flags.overlay_score:      
            frame = self.draw_score(frame)
        return frame
    
    def draw_score(self, frame):
        if len(self.score_history) != 0:
            score = list(self.score_history.values())[-1]
            p1 = score.get("player1", "P1")
            p2 = score.get("player2", "P2")
            set_score = score.get("set_score", [0, 0])
            point_score = score.get("point_score", [0, 0])

            # Display format: Name   Set   Point
            server = self.points.current_server  # 'player1' or 'player2'
            server_name = p1 if server == 'player1' else p2

            # Display lines
            lines = [
                f"{p1}   {set_score[0]}   {self._format_point(point_score[0])}",
                f"{p2}   {set_score[1]}   {self._format_point(point_score[1])}",
                f"Serve: {server_name}"
            ]

            for idx, line in enumerate(lines):
                y_offset = 30 + idx * 30
                cv2.putText(frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame
    
    def _format_point(self, val):
        if val == 50:
            return "AD"
        return str(val)
    
    def _debounce_and_commit_score(self, new_score, ts):
        """
        Debounces a stable score reading:
          - Buffers (new_score, ts) in a time‐window queue.
          - Drops entries older than debounce_secs.
          - If all buffered entries match the current new_score, and the span >= debounce_secs,
            returns True once, then resets.
          - Ignores any new_score that is lower than the last committed score.
        """

        # If it’s the same as our last committed, clear and skip
        last = self._last_committed_score
        if last and new_score['set_score'] == last['set_score'] and new_score['point_score'] == last['point_score']:
            self._score_queue.clear()
            return False
        
        if last:
            new_points = new_score['point_score']
            last_points = last['point_score']

            # Ignore if new points are less than last points (regression)
            if (new_points[0] < last_points[0]) or (new_points[1] < last_points[1]):
                self._score_queue.clear()
                return False

        # Append new reading
        self._score_queue.append((new_score, ts))

        # Drop old entries beyond debounce window
        while self._score_queue and (ts - self._score_queue[0][1] > self.debounce_secs):
            self._score_queue.popleft()

        # Check stability
        stable = all(
            entry[0]['set_score'] == new_score['set_score'] and
            entry[0]['point_score'] == new_score['point_score']
            for entry in self._score_queue
        )
        duration = self._score_queue[-1][1] - self._score_queue[0][1] if self._score_queue else 0

        if stable and duration >= self.debounce_secs:
            # commit once
            self._last_committed_score = {
                'set_score': new_score['set_score'],
                'point_score': new_score['point_score']
            }
            self._score_queue.clear()
            return True

        return False
    
    def _is_valid_score(self, score_data):
        """
        Validate that score data contains valid integers
        """
        set_score = score_data.get('set_score')
        point_score = score_data.get('point_score')
        def is_valid_score_list(score_list):
            return isinstance(score_list, list) and len(score_list) == 2 and all(isinstance(s, int) for s in score_list)
        return is_valid_score_list(set_score) and is_valid_score_list(point_score)
      
    def _draw_highlight_overlay(self, frame, highlights):
        """
        Draw highlight information on the frame
        """
        y_pos = 50
        for highlight_type, detected in highlights.items():
            if detected:
                color = (0, 255, 0) if highlight_type == 'break_point' else (0, 0, 255)
                text = f"{highlight_type.replace('_', ' ').title()} Detected!"
                cv2.putText(frame, text, (95, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                y_pos += 30
    

class PlayerTracker:
    def __init__(self):
        detector_cfg = DetectorConfig(cfg)
        player_config = detector_cfg.get(DetectorType.PLAYER)
        self.detector = Detector(player_config)
        
        self.tracker = Tracker(cfg)
        self.colors = cfg.general.COLORS

    def track_players(self, frame):
        results, frame = self.detector.detect(frame.copy())
        detections = results[0].boxes.data.cpu().numpy()
        
        online_targets = self.tracker.track(frame.copy(), detections)

        if cfg.flags.overlay_track:
            for obj in online_targets:
                x1, y1, x2, y2, track_id = map(int, obj[:5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['green'], 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['green'], 2)
        return online_targets, frame


class ScoreReader:
    def __init__(self):
        detector_cfg = DetectorConfig(cfg)
        scoreboard_config = detector_cfg.get(DetectorType.SCOREBOARD)
        self.detector = Detector(scoreboard_config)
        
        self.OCRinference = PaddleOCRProcessor(lang='en')

    def get_score(self, frame):
        score = None
        results, frame = self.detector.detect(frame)
        
        detections = results[0].boxes.data.cpu().numpy()
        boxes = detections[:, :-2].astype(int) 
        for i, box in enumerate(boxes):
            x, y, x2, y2 = map(int, box)
            w, h = x2 - x, y2 - y
            score_image = frame[y: y + h, x: x + w]
            if cfg.OCR.preprocess:
                score_image = self.OCRinference.preprocess_image(score_image)
            ocr_results, score = self.OCRinference.process_image(score_image)            
            # frame = self.OCRinference.draw_score(frame, ocr_results, [x, y, w, h])
            
        if score:
            p1 = score.get("player1", "P1")
            p2 = score.get("player2", "P2")
            set_score = score.get("set_score", [0, 0])
            point_score = score.get("point_score", [0, 0])

            # Display format: Name   Set   Point
            lines = [
                f"{p1}   {set_score[0]}   {self._format_point(point_score[0])}",
                f"{p2}   {set_score[1]}   {self._format_point(point_score[1])}"
            ]

            for idx, line in enumerate(lines):
                y_offset = 250 + idx * 30
                cv2.putText(frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return frame, score
    
    def _format_point(self, val):
        if val == 50:
            return "AD"
        return str(val)
    
class Points:
    """
    Detects tennis highlights based on score patterns:
      - Game-ending points (when a game concludes)
      - Break points (when receiver has a chance to break serve at 30-40)
      - Advantage points separately when score enters advantage state
    Usage:
        points = Points()
        highlights = points.process(timestamp, score_data)
    """
    def __init__(self):
        # Last seen timestamp and score
        self.last_score = None  # tuple(timestamp, score_data)
        # Current server: 'player1' or 'player2'
        self.current_server = 'player1'
        # Histories
        self.game_ending_points = []
        self.break_points = []
        self.advantage_points = []

    def process(self, timestamp, score):
        """
        Process a new score update and detect highlights.

        Args:
            timestamp (float): Video timestamp of the score.
            score (dict): {
                'player1': str, 'player2': str,
                'set_score': [int, int], 'point_score': [int, int]
            }

        Returns:
            dict: flags for detected highlights, e.g.
                  {'game_ending': True, 'break_point': True, 'advantage': True}
        """
        highlights = {}

        # Initialize on first reading
        if self.last_score is None:
            self.last_score = (timestamp, score)
            return highlights

        prev_ts, prev = self.last_score
        curr = score

        # 1) Game end detection
        if self._is_game_end(prev, curr):
            highlights['game_ending'] = True
            self._record_game_end(prev_ts, prev)
            self._swap_server()

        # 2) Break point detection (receiver at 40-30)
        if self._is_break_point(curr):
            highlights['break_point'] = True
            self._record_break_point(prev_ts, prev)

        # 3) Advantage detection (either player on AD after deuce)
        if self._is_advantage(curr):
            highlights['advantage'] = True
            self._record_advantage_point(prev_ts, prev)

        # Update state
        self.last_score = (timestamp, curr)
        return highlights

    def _is_game_end(self, prev, curr):
        prev_pt = prev['point_score']
        curr_pt = curr['point_score']
        # a) Points reset -> new game
        if curr_pt == [0, 0] and any(p > 0 for p in prev_pt):
            return True
        # b) Set score increment
        if prev['set_score'] != curr['set_score']:
            return True
        return False

    def _is_break_point(self, curr):
        # Only at 30-40 (receiver holds break chance)
        pt = curr['point_score']
        if self.current_server == 'player1':
            # Receiver is player2 at break point
            return pt == [30, 40]
        else:
            return pt == [40, 30]

    def _is_advantage(self, curr):
        # Advantage only after deuce (both had 40)
        pt = curr['point_score']
        # AD for player1: [>40, 40]
        # AD for player2: [40, >40]
        return (pt[0] > 40 and pt[1] == 40) or (pt[1] > 40 and pt[0] == 40)

    def _record_game_end(self, timestamp, score):
        record = {
            'timestamp': timestamp,
            'player1': score.get('player1'),
            'player2': score.get('player2'),
            'set_score': tuple(score.get('set_score', [0,0])),  # (p1_sets, p2_sets)
            'point_score': tuple(score.get('point_score', [0,0]))
        }
        self.game_ending_points.append(record)

    def _record_break_point(self, timestamp, score):
        record = {
            'timestamp': timestamp,
            'player1': score.get('player1'),
            'player2': score.get('player2'),
            'set_score': tuple(score.get('set_score', [0,0])),  # (p1_sets, p2_sets)
            'point_score': tuple(score.get('point_score', [0,0])),  # (p1_pts, p2_pts)
            'server': self.current_server,
            'receiver': 'player2' if self.current_server == 'player1' else 'player1'
        }
        self.break_points.append(record)

    def _record_advantage_point(self, timestamp, score):
        record = {
            'timestamp': timestamp,
            'player1': score.get('player1'),
            'player2': score.get('player2'),
            'set_score': tuple(score.get('set_score', [0,0])),  # (p1_sets, p2_sets)
            'point_score': tuple(score.get('point_score', [0,0])),
            'server': self.current_server
        }
        self.advantage_points.append(record)

    def _swap_server(self):
        self.current_server = 'player2' if self.current_server == 'player1' else 'player1'

    def get_highlights(self):
        return {
            'game_ending_points': self.game_ending_points,
            'break_points': self.break_points,
            'advantage_points': self.advantage_points
        }
        

if __name__ == "__main__":
    scorer = ScoreReader()
    source = 'tk2.png'
    image = cv2.imread(source)
    frame, new_score = scorer.get_score(image)
    print(new_score)



