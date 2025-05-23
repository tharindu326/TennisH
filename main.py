import os
import cv2
from config import cfg
import numpy as np
import time
import cv2
from datetime import datetime
import os
import time
from tqdm import tqdm
from highlights import Highlights
import argparse

script_dir = os.path.dirname(os.path.realpath(__file__))

'''
USAGE: python main.py -s test.mp4
'''
        
class Video:
    def __init__(self, videoPath, Vidout_path=cfg.general.output_path):
        self.cap = cv2.VideoCapture(videoPath)
        self.current_time_stamp = datetime.now()
        self.writer = None
        self.time_array = []
        self.Vidout_path = Vidout_path
        os.makedirs(Vidout_path, exist_ok=True)

        # Determine source identifier
        path_prefix = ''.join(letter for letter in str(videoPath).split(':')[0] if letter.isalnum())
        if path_prefix == 'rtsp':
            print(f'start processing RTSP stream {videoPath}')
            source_name = ".".join(videoPath.split('@')[-1].split('.')[:-1]).replace(':', '-').replace('/', '_')
            self.source_identifier = f'{source_name}'
            self.total_frames = None
        elif isinstance(videoPath, int):
            print(f'start processing camera device {videoPath}')
            self.source_identifier = f'CAM_device_{videoPath}'
            self.total_frames = None
        else:
            print(f'start processing video {videoPath}')
            # source_name = ".".join(videoPath.split('/')[-1].split('.')[:-1])
            source_name = os.path.splitext(os.path.basename(videoPath))[0]
            self.source_identifier = f'{source_name}'
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames = frame_count if frame_count > 0 else None

        self.outVideoPath = os.path.join(Vidout_path, f'processed_{self.source_identifier}.mp4')
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.video_fps > 0:
            print("Couldn't retrieve FPS ..")
            if isinstance(cfg.video.FPS, int):
                self.video_fps = cfg.video.FPS
                print(f'Reading the stream FPS as {self.video_fps} from the config')
            else:
                raise ValueError('Source FPS is not defined correctly or undefined. Please define it in cfg.video.FPS')
        self.highlights = Highlights(videoPath, self.video_fps)
        print(f'video_fps: {self.video_fps}')

    @staticmethod
    def select_frames(camera_fps, required_fps):
        if required_fps is None:
            return [int(i) for i in np.arange(0, camera_fps, dtype=float)]
        delta = camera_fps - 1
        step = delta / required_fps
        y = np.arange(0, required_fps, dtype=float) * step + 1
        return [int(i) for i in y]

    def run(self):
        # Setup progress bar for video files only
        pbar = None
        if self.total_frames:
            pbar = tqdm(total=self.total_frames, desc=f'Processing {self.source_identifier}')

        selected_frameIDs = self.select_frames(camera_fps=self.video_fps,
                                               required_fps=cfg.video.requiredFPS)
        count = 0
        frame_filter_Count = 1

        while True:
            ret, frame = self.cap.read()
            video_timestamp = round(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)

            if not ret:
                break

            if pbar:
                pbar.update(1)

            if cfg.general.frame_resize:
                frame = cv2.resize(frame, cfg.general.FrameSize)
            if cfg.general.frame_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            start_time = time.time()
            if frame_filter_Count in selected_frameIDs:
                
                frame = self.highlights.generate(frame, video_timestamp)
                
                if cfg.flags.image_show:
                    h, w = frame.shape[:2]
                    frame_show = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    cv2.imshow(self.source_identifier, frame_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if cfg.video.save:
                    if self.writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*cfg.video.FOURCC)
                        self.writer = cv2.VideoWriter(self.outVideoPath, fourcc, cfg.video.requiredFPS,
                                                      (frame.shape[1], frame.shape[0]), True)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (255, 0, 0)
                    thickness = 2
                    # Overlay frame number top-right
                    cv2.putText(frame, f'Frame: {int(count)}',
                                (frame.shape[1] - 280, 30), font,
                                cfg.general.TextSize, color, thickness, lineType=cv2.LINE_AA)

                    self.writer.write(frame)

                count += 1

            # Update frame filter and timing
            frame_filter_Count = frame_filter_Count + 1 if frame_filter_Count < self.video_fps else 1
            elapsed = time.time() - start_time
            self.time_array.append(elapsed)
            if cfg.flags.render_fps:
                fps_print = len(self.time_array) / sum(self.time_array)
                print(f'Frame: {int(count)} | FPS: {fps_print:.2f}')
            if len(self.time_array) > 30:
                self.time_array.pop(0)

        if pbar:
            pbar.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='video path source(s)')
    args = parser.parse_args()
    video_instance = Video(videoPath=args.source)
    video_instance.run()
    