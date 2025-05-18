import cv2
from Yolo.detect import Detector
from tracker import Tracker
from config import cfg, DetectorConfig, DetectorType
from OCR.paddle_OCR import PaddleOCRProcessor
from collections import deque
import csv
import os 
from datetime import datetime
import numpy as np
import re


class Score:
    def __init__(self, start_time, end_time, score, energy, events):
        self.start_time = start_time
        self.end_time = end_time
        self.energy = energy
        self.events = events
        self.s1, self.s2 = score['set_score']
        self.p1, self.p2 = score['point_score']
        self.player1 = score.get('player1')
        self.player2 = score.get('player2')   
        

class Highlight():
    def __init__(self, score_obj: Score, h_types):
        self.htypes = h_types
        self.score_obj = score_obj
        self.start_time = score_obj.start_time
        self.end_time = score_obj.end_time
        
        
class Highlights:
    def __init__(self, input_video_path):
        self.tracker = PlayerTracker()
        self.scorer = ScoreReader()
        self.scores = []
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
                'start_time',
                'end_time',
                'event',
                'player',
                'set_score',
                'point_score',
                'energy'
            ])
        
        # Points
        # Last seen timestamp and score
        self.last_score = None  # tuple(timestamp, score_data)
        # Current server: 'player1' or 'player2'
        self.current_server = 'player1'

    def player_track(self, frame, video_timestamp):
        """Track players in the frame"""
        return self.tracker.track_players(frame, video_timestamp)

    def get_score(self, frame, video_timestamp): 
        """
        Process the frame to extract score and detect break/end points
        """
        frame, new_score = self.scorer.get_score(frame)
        if not new_score:
            return frame
        if not self._is_valid_score(new_score):
            return frame
        
        highlights = {}
        
        if self._debounce_and_commit_score(new_score, video_timestamp):
            scoring_events = self.process_score(video_timestamp, new_score)
           
            p1, p2 = self.tracker._top_two_players()
            energy = p1.compute_total_distance() + p2.compute_total_distance()
            # flag extended rally if movement energy exceeds threshold
            if energy >= cfg.highlights.extended_rally_energy_threshold:
                highlights['extended_rally'] = True
            
            # remove the players after a point
            self.tracker.players = []

            self.scores.append(Score(start_time=min(p1.first_seen, p2.first_seen), end_time=max(p1.last_seen, p2.last_seen), score=new_score, 
                               energy=energy, events=list(scoring_events.keys())))
            
            for k, v in scoring_events.items():
                if v and k in ['advantage', 'break_point', 'game_ending']:
                    highlights[k] = True
                    
            self.highlights.append(Highlight(score_obj=self.scores[-1], h_types=highlights))
            
            if highlights:
                if cfg.flags.overlay_highlights:
                    self._draw_highlight_overlay(frame, highlights)

            self.score2CSV()
            
        if cfg.flags.overlay_score:      
            frame = self.draw_score(frame)
        return frame
    
    def process_score(self, timestamp, score):
        """
        Process a new score update and detect events.
            - Game-ending points (when a game concludes)
            - Break points (when receiver has a chance to break serve at 30-40)
            - Advantage points separately when score enters advantage state
        """
        scoring_events = {}

        # Initialize on first reading
        if self.last_score is None:
            self.last_score = (timestamp, score)
            return scoring_events

        prev_ts, prev = self.last_score
        curr = score
        
        # Set end detection
        if self._is_set_end(curr):
            scoring_events['set_ending'] = True

        # Game end detection
        if self._is_game_end(prev, curr):
            scoring_events['game_ending'] = True
            self._swap_server()

        # Break point detection (receiver at 40-30)
        if self._is_break_point(curr):
            scoring_events['break_point'] = True

        # Advantage detection (either player on AD after deuce)
        if self._is_advantage(curr):
            scoring_events['advantage'] = True
        self.last_score = (timestamp, curr)
        return scoring_events

    def _is_game_end(self, prev, curr):
        prev_pt = prev['point_score']
        curr_pt = curr['point_score']
        # Points reset -> new game
        if curr_pt == [0, 0] and any(p > 0 for p in prev_pt):
            return True
        # Set score increment
        if prev['set_score'][0] < curr['set_score'][0] or prev['set_score'][1] < curr['set_score'][1]:
            return True
        return False
    
    def _is_set_end(self, curr):
        curr_pt = curr['point_score']
        curr_set = curr['set_score']
        # Points reset -> new game
        if curr_pt == [0, 0] and curr_set == [0, 0]:
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
    
    def _swap_server(self):
        self.current_server = 'player2' if self.current_server == 'player1' else 'player1'
    
    def score2CSV(self):
        score = self.scores[-1]

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # player 1 row
            writer.writerow([
                score.start_time,
                score.end_time,
                ','.join(score.events),
                score.player1,
                score.s1,
                score.p1,
                score.energy
            ])
            writer.writerow([
                '',
                '',
                '',
                score.player2,
                score.s2,
                score.p2,
                ''
            ])
        
    def draw_score(self, frame):
        if len(self.scores) != 0:
            score = self.scores[-1]
            # Display format: Name   Set   Point
            server = self.current_server  # 'player1' or 'player2'
            server_name = score.player1 if server == 'player1' else score.player2

            # Display lines
            lines = [
                f"{score.player1}   {score.s1}   {self._format_point(score.p1)}",
                f"{score.player2}   {score.s2}   {self._format_point(score.p2)}",
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
        Debounces a stable score reading, with an optional “easy AD” override:
        - Buffers (new_score, ts) in a time‐window queue.
        - Drops entries older than debounce_secs.
        - If easy_AD is enabled and the score is AD vs. 40 ([50,40] or [40,50]),
            commits as soon as it’s been present for easy_AD_pct * debounce_secs.
        - Otherwise, waits until all buffered entries match new_score for full debounce_secs.
        - Ignores any new_score that is lower than the last committed score.
        Returns True once when the score is committed, else False.
        """
        last = self._last_committed_score
        # Skip identical to last commit
        if last and new_score['set_score'] == last['set_score'] and new_score['point_score'] == last['point_score']:
            self._score_queue.clear()
            return False

        # Skip regressions
        if last:
            new_pts = new_score['point_score']
            last_pts = last['point_score']
            if new_pts[0] < last_pts[0] or new_pts[1] < last_pts[1]:
                self._score_queue.clear()
                return False

        # Buffer this reading
        self._score_queue.append((new_score, ts))
        # Evict old readings
        while self._score_queue and (ts - self._score_queue[0][1] > self.debounce_secs):
            self._score_queue.popleft()

        # Compute duration in buffer
        duration = self._score_queue[-1][1] - self._score_queue[0][1]

        # Easy-AD
        pts = new_score['point_score']
        is_ad_case = (pts == [50, 40]) or (pts == [40, 50])
        if cfg.highlights.easy_AD and is_ad_case:
            threshold = self.debounce_secs * cfg.highlights.easy_AD_percentage
            if duration >= threshold:
                self._last_committed_score = {
                    'set_score': new_score['set_score'],
                    'point_score': new_score['point_score']
                }
                self._score_queue.clear()
                return True

        # Full debounce
        stable = all(
            entry[0]['set_score'] == new_score['set_score'] and
            entry[0]['point_score'] == new_score['point_score']
            for entry in self._score_queue
        )
        if stable and duration >= self.debounce_secs:
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

        self.players = []
        self._counters = {}

    def _find_player_by_id(self, track_id):
        """
        Retrieve a Player object by track_id, or None if not found.
        """
        for player in self.players:
            if player.id == track_id:
                return player
        return None

    def track_players(self, frame, timestamp):
        """
        Detect and track players in the given frame, confirming new players
        after enough appearances, updating existing ones, and pruning lost players.
        """
        results, vis_frame = self.detector.detect(frame.copy())
        detections = results[0].boxes.data.cpu().numpy()
        online_targets = self.tracker.track(vis_frame.copy(), detections)

        current_time = timestamp
        for obj in online_targets:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            bbox = (x1, y1, x2, y2)

            player = self._find_player_by_id(track_id)
            if player is None:
                count = self._counters.get(track_id, 0) + 1
                self._counters[track_id] = count
                if count >= cfg.players.confirm_threshold:
                    player = Player(track_id, bbox, timestamp)
                    self.players.append(player)
                    del self._counters[track_id]
            else:
                player.update(bbox, timestamp)
            if cfg.flags.overlay_track:
                color = self.colors['green'] if player else self.colors['yellow']
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f'ID:{track_id}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Remove players not seen within the timeout
        # self.players = [p for p in self.players if (current_time - p.last_seen) <= cfg.players.deactivation_timeout]
        return online_targets, vis_frame
    
    def _top_two_players(self):
        """
        Return the two Player objects with the longest track history whose
        seen-intervals overlap in time. Pairs are tested in index-order:
        (1,2), (1,3), (2,3), (1,4), (2,4), (3,4), ... up to (n-1,n).
        If no overlapping pair exists, fall back to the two longest tracks.
        If fewer than two players exist, return (p, p) or (None, None).
        """
        players = self.players
        num_players = len(players)

        if num_players == 0:
            return None, None
        if num_players == 1:
            return players[0], players[0]

        player_data = [
            (
                player,
                player.get_track_length(),
                player.first_seen,
                player.last_seen,
            )
            for player in players
        ]
        # Sort descending by track length
        player_data.sort(key=lambda item: item[1], reverse=True)

        # Test pairs in index-order: for j in 1..num_players-1, for i in 0..j-1
        for j in range(1, num_players):
            player_b, _, start_b, end_b = player_data[j]
            for i in range(j):
                player_a, _, start_a, end_a = player_data[i]
                if self._intervals_overlap(start_a, end_a, start_b, end_b):
                    return player_a, player_b

        # Fallback to the two longest tracks
        top_two = player_data[:2]
        return top_two[0][0], top_two[1][0]


    @staticmethod
    def _intervals_overlap(start1, end1, start2, end2):
        return start1 <= end2 and start2 <= end1


class ScoreReader:
    def __init__(self):
        detector_cfg = DetectorConfig(cfg)
        scoreboard_config = detector_cfg.get(DetectorType.SCOREBOARD)
        self.detector = Detector(scoreboard_config)
        
        use_gpu = cfg.general.device != 'cpu'
        self.OCRinference = PaddleOCRProcessor(lang='en', use_gpu=use_gpu)

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
            result = self.OCRinference.ocr.ocr(score_image, cls=True)
            ocr_results, score = self.process_score(result=result)
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
    
    def process_score(self, result):
        ret = {"boxes": [], "texts": [], "scores": []}
        score = None
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
    

class Player:
    """
    Represents a tracked player, stores its ID, history of bounding boxes, and last seen time.
    Provides movement-processing utilities.
    """
    def __init__(self, track_id, initial_bbox, timestamp, max_history=30):
        self.id = track_id
        self.bboxes = deque()
        self.bboxes.append(initial_bbox)
        self._appearances = 1
        # Timestamp of the last update (in seconds)
        self.first_seen = timestamp
        self.last_seen = timestamp

    def update(self, bbox, timestamp):
        """
        Update the player's bounding box history and last-seen timestamp.
        """
        self.bboxes.append(bbox)
        self._appearances += 1
        self.last_seen = timestamp
        
    def get_track_length(self):
        return len(self.bboxes)

    @property
    def appearance_count(self):
        return self._appearances

    @property
    def latest_bbox(self):
        return self.bboxes[-1] if self.bboxes else None

    def get_trajectory(self):
        """
        Return the list of all stored bounding boxes for trajectory analysis.
        """
        return list(self.bboxes)

    def get_centroids(self):
        """
        Compute centroid points from stored bounding boxes.
        """
        centroids = []
        for x1, y1, x2, y2 in self.bboxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroids.append((cx, cy))
        return centroids
    
    def compute_total_distance(self):
        """
        Sum the Euclidean distance between consecutive centroids.
        """
        centroids = self.get_centroids()
        if len(centroids) < 2:
            return 0.0
        dist = 0.0
        for (x0, y0), (x1, y1) in zip(centroids, centroids[1:]):
            dist += ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        return dist
    

if __name__ == "__main__":
    scorer = ScoreReader()
    source = 'tk2.png'
    image = cv2.imread(source)
    frame, new_score = scorer.get_score(image)
    print(new_score)



