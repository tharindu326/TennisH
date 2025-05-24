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
import copy
from enum import Enum


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


class CourtSide(Enum):
    TOP = 0
    BOTTOM = 1
    UNKNOWN = 2
    

class Ball:
    def __init__(self, initial_bbox, timestamp, net_box, max_history=30):
        self.bboxes = deque()
        self.bboxes.append(initial_bbox)
        self.timestamps = deque()
        self.timestamps.append(timestamp)
        self.nets = deque()
        self.nets.append(net_box)
        self.first_seen = timestamp
        self.last_seen = timestamp
        
    def get_ball_track(self, start_time, end_time):
        trajectory = [(bbox, self.timestamps[i], self.nets[i]) for i, bbox in enumerate(self.bboxes) if start_time <= self.timestamps[i] <= end_time]
        return trajectory
        
    def update(self, bbox, timestamp, net_box):    
        self.bboxes.append(bbox)
        self.timestamps.append(timestamp)
        self.nets.append(net_box)
        self.last_seen = timestamp

    def get_aces_serviceWinners(self, start_time, end_time):
        trajectory = self.get_ball_track(start_time, end_time)
        
        hits = 0
        returns = 0
        current_rally_hits = 0
        direction_changes = []
        velocities = []
        current_side = CourtSide.UNKNOWN

        prev_centroid = None
        prev_timestamp = None

        for i, (bbox, timestamp, net) in enumerate(trajectory):
            centroid = self._get_centroid(bbox)

            if prev_centroid is not None and prev_timestamp is not None:
                dt = timestamp - prev_timestamp
                if dt > 0:
                    vx = (centroid[0] - prev_centroid[0]) / dt
                    vy = (centroid[1] - prev_centroid[1]) / dt
                    velocities.append((vx, vy))

                    if len(velocities) >= 2:
                        prev_vx, prev_vy = velocities[-2]
                        curr_vx, curr_vy = velocities[-1]
                        dot_product = prev_vx * curr_vx + prev_vy * curr_vy
                        prev_mag = (prev_vx**2 + prev_vy**2)**0.5
                        curr_mag = (curr_vx**2 + curr_vy**2)**0.5

                        if prev_mag > 0 and curr_mag > 0:
                            cosine = dot_product / (prev_mag * curr_mag)
                            cosine = max(min(cosine, 1.0), -1.0)
                            angle = np.arccos(cosine) * 180 / np.pi

                            if angle > 45:
                                hits += 1
                                current_rally_hits += 1
                                direction_changes.append(i)

            new_side = self._determine_side(bbox, net_box=net)
            if new_side != current_side and \
            new_side != CourtSide.UNKNOWN and \
            current_side != CourtSide.UNKNOWN:
                returns += 1

            current_side = new_side
            prev_centroid = centroid
            prev_timestamp = timestamp

        # Ace: ball crossed net once, no return hit detected
        is_ace = (returns == 1 and current_rally_hits == 0)

        # Server winner: rally with one hit and one return crossing
        is_server_winner = (current_rally_hits == 1 and returns == 1)

        return is_ace, is_server_winner
    
    def _get_centroid(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return (cx, cy)
    
    def _determine_side(self, bbox, net_box):
        if not net_box:
            return CourtSide.UNKNOWN
        ball_centroid = self._get_centroid(bbox)
        net_y = (net_box[1] + net_box[3]) / 2
        if ball_centroid[1] < net_y:
            return CourtSide.TOP
        else:
            return CourtSide.BOTTOM
    

class Highlight():
    def __init__(self, score_obj: Score, h_types):
        self.htypes = h_types
        self.score_obj = score_obj
        self.start_time = score_obj.start_time
        self.end_time = score_obj.end_time
        
        
class Highlights:
    def __init__(self, input_video_path, source_fps):
        self.detector = Detector()
        self.tracker = GameTracker()
        self.scorer = ScoreReader()
        self.ball = None
        self.scores = []
        self.highlights = []
        self.maxlen = int(source_fps * cfg.highlights.debounce_secs_for_score_change)
        self._score_queue = deque(maxlen=self.maxlen)
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
        self.start_flag = True

    def generate(self, frame, video_timestamp):
        """Track players in the frame"""
        
        results, frame_det = self.detector.detect(frame.copy())
        detections = results[0].boxes.data.cpu().numpy()
        online_targets_player, frame_players = self.tracker.track_players(frame_det, video_timestamp, detections)
        
        # update ball data
        class_ids = detections[:, -1].astype(int)
        ball_dets = detections[(class_ids == 0)] # ball
        ball_boxes = ball_dets[:, :-2].astype(int)
        net_dets = detections[(class_ids == 0)] # net
        net_boxes = net_dets[:, :-2].astype(int)
        if len(net_boxes) > 0:
            net = net_boxes[0].tolist()
        else:
            net = []
        for box in ball_boxes:
            x1, y1, x2, y2 = box
            bbox = (x1, y1, x2, y2)
            if self.ball is None:
                self.ball = Ball(bbox, video_timestamp, net) 
            else:
                self.ball.update(bbox, video_timestamp, net_box=net)
        
        # if cfg.tracker.ball_track:
        #     online_targets_ball, frame_out = self.tracker.track_ball(frame_players, video_timestamp, detections)
        # else:
        #     online_targets_ball, frame_out = [], frame_players
        
        score_box_dets = detections[(class_ids == 4)] # score
        boxes = score_box_dets[:, :-2].astype(int)
        frame_score = self.get_score(frame_det, video_timestamp, boxes, net)
        
        return frame_score
        

    def get_score(self, frame, video_timestamp, boxes, net): 
        """
        Process the frame to extract score and detect break/end points
        """
        frame, new_score = self.scorer.get_score(frame, boxes)
        if not new_score:
            return frame
        if not self._is_valid_score(new_score):
            return frame
        
        highlights = {}
        
        if self._debounce_and_commit_score(new_score, video_timestamp):
            scoring_events = self.process_score(video_timestamp, new_score)
           
            p1, p2 = self.tracker._top_two_players(net)
            if p1 and p2:
                score_start_time = min(p1.first_seen, p2.first_seen)
                score_end_time = max(p1.last_seen, p2.last_seen)
                
                # get the ball tragectory by time 
                is_ace, is_server_winner = self.ball.get_aces_serviceWinners(score_start_time, score_end_time)
                if is_ace:
                    highlights['ace'] = True
                if is_server_winner:
                    highlights['service_winner'] = True
                    
                energy = p1.compute_total_distance() + p2.compute_total_distance()
                # flag extended rally if movement energy exceeds threshold
                if energy >= cfg.highlights.extended_rally_energy_threshold:
                    highlights['extended_rally'] = True
                
                # remove the players after a point
                self.tracker.players = []

                self.scores.append(Score(start_time=score_start_time, end_time=score_end_time, score=new_score, 
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
    
    def _get_possible_next_scores(self, current_score):
        """
        Given current score, return list of logically possible next score states.
        Returns list of dicts with 'point_score' and 'set_score' keys.
        """
        if not current_score:
            return []
        
        curr_pts = current_score['point_score']
        curr_sets = current_score['set_score']
        possible_scores = []
        
        p1_pts, p2_pts = curr_pts
        s1_sets, s2_sets = curr_sets
        
        # Helper to create score dict
        def make_score(p1, p2, s1=None, s2=None):
            return {
                'point_score': [p1, p2],
                'set_score': [s1 if s1 is not None else s1_sets, 
                            s2 if s2 is not None else s2_sets]
            }
        
        # Regular scoring (0, 15, 30)
        if p1_pts < 30 and p2_pts < 30:
            possible_scores.extend([
                make_score(p1_pts + 15, p2_pts),  # Player 1 scores
                make_score(p1_pts, p2_pts + 15)   # Player 2 scores
            ])
        
        # One player at 30, other below 30
        elif p1_pts == 30 and p2_pts < 30:
            possible_scores.extend([
                make_score(40, p2_pts),           # Player 1 to 40
                make_score(p1_pts, p2_pts + 15)  # Player 2 scores
            ])
        
        elif p1_pts < 30 and p2_pts == 30:
            possible_scores.extend([
                make_score(p1_pts + 15, p2_pts),  # Player 1 scores
                make_score(p1_pts, 40)            # Player 2 to 40
            ])
        
        # Both at 30 (30-30)
        elif p1_pts == 30 and p2_pts == 30:
            possible_scores.extend([
                make_score(40, 30),  # Player 1 to 40
                make_score(30, 40)   # Player 2 to 40
            ])
        
        # Game point scenarios (40 vs less than 40, but not deuce)
        elif p1_pts == 40 and p2_pts < 40:
            if p2_pts == 30:
                # Special case: 40-30, player 2 can tie to deuce
                possible_scores.extend([
                    make_score(0, 0, s1_sets + 1, s2_sets),  # Player 1 wins game
                    make_score(40, 40)                        # Player 2 ties to deuce
                ])
            else:
                # 40-0 or 40-15
                possible_scores.extend([
                    make_score(0, 0, s1_sets + 1, s2_sets),  # Player 1 wins game
                    make_score(p1_pts, p2_pts + 15)          # Player 2 scores
                ])
        
        elif p1_pts < 40 and p2_pts == 40:
            if p1_pts == 30:
                # Special case: 30-40, player 1 can tie to deuce
                possible_scores.extend([
                    make_score(0, 0, s1_sets, s2_sets + 1),  # Player 2 wins game
                    make_score(40, 40)                        # Player 1 ties to deuce
                ])
            else:
                # 0-40 or 15-40
                possible_scores.extend([
                    make_score(0, 0, s1_sets, s2_sets + 1),  # Player 2 wins game
                    make_score(p1_pts + 15, p2_pts)          # Player 1 scores
                ])
        
        # Deuce and advantage scenarios
        elif p1_pts == 40 and p2_pts == 40:  # Deuce
            possible_scores.extend([
                make_score("AD", 0),  # Player 1 advantage
                make_score(0, "AD")   # Player 2 advantage
            ])
        
        elif p1_pts == "AD" and p2_pts == 40:  # Player 1 advantage
            possible_scores.extend([
                make_score(0, 0, s1_sets + 1, s2_sets),  # Player 1 wins game
                make_score(40, 40)                       # Back to deuce
            ])
        
        elif p1_pts == 40 and p2_pts == "AD":  # Player 2 advantage
            possible_scores.extend([
                make_score(0, 0, s1_sets, s2_sets + 1),  # Player 2 wins game
                make_score(40, 40)                       # Back to deuce
            ])
        
        return possible_scores

    def _is_predicted_score(self, new_score):
        """
        Check if new_score matches one of the logically possible next states.
        """
        if not self._last_committed_score:
            return False
        
        possible_scores = self._get_possible_next_scores(self._last_committed_score)
        # print("possible_scores", possible_scores)
        # print("new_scores", new_score)
        for possible in possible_scores:
            if (new_score['point_score'] == possible['point_score'] and 
                new_score['set_score'] == possible['set_score']):
                return True
        
        return False

    def _debounce_and_commit_score(self, new_score, ts):
        """
        Predictive debouncing using fixed-size deque:
        - Deque automatically drops old values when maxlen is reached
        - Accept any valid score initially when no committed score exists
        - Only accept logically expected scores afterward
        - Commit when deque is full and all entries are stable
        """
        last = self._last_committed_score
        
        if self.start_flag:
            if new_score['set_score'] == [0, 0] and new_score['point_score'] == [0, 0]:
                self.start_flag = False  # Game has started
            else:
                # Ignore everything until 0-0 appears
                return False
                
        # Skip identical to last commit
        # if last and new_score['set_score'] == last['set_score'] and new_score['point_score'] == last['point_score']:
        #     self._score_queue.clear()
        #     return False

        # Skip regressions (only if we have a last committed score)
        # if last:
        #     new_pts = new_score['point_score']
        #     last_pts = last['point_score']
        #     if new_pts[0] < last_pts[0] or new_pts[1] < last_pts[1]:
        #         self._score_queue.clear()
        #         return False

        # If no committed score yet, accept any valid score (initial state)
        # Otherwise, only accept predicted/expected scores
        if last and not self._is_predicted_score(new_score):
            self._score_queue.clear()
            return False

        # Add to deque - old values automatically dropped when maxlen reached
        self._score_queue.append((new_score, ts))
        
        # Check if deque is full and all entries are stable
        if len(self._score_queue) == self._score_queue.maxlen:
            stable = all(
                entry[0]['set_score'] == new_score['set_score'] and
                entry[0]['point_score'] == new_score['point_score']
                for entry in self._score_queue
            )
            
            if stable:
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
    

class GameTracker:
    def __init__(self):
        player_cfg = copy.deepcopy(cfg)
        player_cfg.tracker.classes = [3]
        self.tracker_player = Tracker(player_cfg)

        if cfg.tracker.ball_track:
            ball_cfg = copy.deepcopy(cfg)
            ball_cfg.tracker.classes = [0]
            self.tracker_ball = Tracker(ball_cfg)
            
        self.colors = cfg.general.COLORS

        self.players = []
        self._counters = {}
        self.balls = []
        
    def _find_player_by_id(self, track_id):
        """
        Retrieve a Player object by track_id, or None if not found.
        """
        for player in self.players:
            if player.id == track_id:
                return player
        return None
    
    def _find_ball_by_id(self, track_id):
        """
        Retrieve a ball object by track_id, or None if not found.
        """
        for ball in self.balls:
            if ball.id == track_id:
                return ball
        return None        
        
    def track_ball(self, frame, timestamp, detections):
        online_targets_ball = self.tracker_ball.track(frame.copy(), detections)
        for obj in online_targets_ball:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            bbox = (x1, y1, x2, y2)

            if cfg.flags.overlay_ball_track:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['blue'], 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['blue'], 2)

        # Remove players not seen within the timeout
        # self.players = [p for p in self.players if (current_time - p.last_seen) <= cfg.players.deactivation_timeout]
        return online_targets_ball, frame

    def track_players(self, frame, timestamp, detections):
        """
        Detect and track players in the given frame, confirming new players
        after enough appearances, updating existing ones, and pruning lost players.
        """
        online_targets_player = self.tracker_player.track(frame.copy(), detections)
        for obj in online_targets_player:
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
            if cfg.flags.overlay_player_track:
                color = self.colors['green'] if player else self.colors['yellow']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Remove players not seen within the timeout
        # self.players = [p for p in self.players if (current_time - p.last_seen) <= cfg.players.deactivation_timeout]
        return online_targets_player, frame

    def _top_two_players(self, net_box=None):
        """
        Return the two Player objects with the longest track history whose
        seen-intervals overlap in time and that are on opposite sides of the net.
        Pairs are tested in index-order: (1,2), (1,3), (2,3), ... up to (n-1,n).
        If no pair both overlaps and straddles the net, fall back to the first overlapping pair.
        If still none, fall back to the two longest tracks.
        If fewer than two players exist, return (p, p) or (None, None).
        """
        players = self.players
        num = len(players)
        if num == 0:
            return None, None
        if num == 1:
            return players[0], players[0]

        # Prepare data: (player, track_length, first_seen, last_seen, centroid_x)
        data = []
        for p in players:
            length = p.get_track_length()
            start, end = p.first_seen, p.last_seen
            x1, y1, x2, y2 = p.last_bbox()
            centroid_x = (x1 + x2) / 2
            data.append((p, length, start, end, centroid_x))
        # Sort descending by track length
        data.sort(key=lambda t: t[1], reverse=True)

        def on_opposite_sides(a_x, b_x, net):
            if net is None or len(net) < 4:
                return True
            nx1, _, nx2, _ = net
            net_center = (nx1 + nx2) / 2
            return (a_x < net_center < b_x) or (b_x < net_center < a_x)

        # First pass: overlap + opposite sides
        for j in range(1, num):
            pb, _, sb, eb, bx = data[j]
            for i in range(j):
                pa, _, sa, ea, ax = data[i]
                if self._intervals_overlap(sa, ea, sb, eb) and on_opposite_sides(ax, bx, net_box):
                    return pa, pb

        # Second pass: any overlapping pair (ignoring net)
        for j in range(1, num):
            pb, _, sb, eb, _ = data[j]
            for i in range(j):
                pa, _, sa, ea, _ = data[i]
                if self._intervals_overlap(sa, ea, sb, eb):
                    return pa, pb

        # Fallback: two longest
        p1, _, _, _, _ = data[0]
        p2, _, _, _, _ = data[1]
        return p1, p2

    @staticmethod
    def _intervals_overlap(start1, end1, start2, end2):
        return start1 <= end2 and start2 <= end1


class ScoreReader:
    def __init__(self):
        use_gpu = cfg.general.device != 'cpu'
        self.OCRinference = PaddleOCRProcessor(lang='en', use_gpu=use_gpu)

    def get_score(self, frame, boxes):
        score = None
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

    def last_bbox(self):
        return self.bboxes[-1]
    
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



