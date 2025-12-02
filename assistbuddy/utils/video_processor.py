"""
Video Processor for camera feeds and video files
Supports frame extraction, scene detection, and activity summarization
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import timedelta
from dataclasses import dataclass


@dataclass
class VideoFrame:
    """Represents a video frame"""
    frame_number: int
    timestamp: str  # HH:MM:SS format
    frame: np.ndarray
    is_keyframe: bool = False


@dataclass
class SceneChange:
    """Represents a scene change"""
    frame_number: int
    timestamp: str
    change_score: float


class VideoProcessor:
    """
    Process video files and camera streams
    Extract keyframes, detect scenes, track activities
    """
    
    def __init__(self):
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.duration_seconds = 0
    
    def load_video(self, video_path: str) -> bool:
        """
        Load video file or camera stream
        
        Args:
            video_path: Path to video file or camera index (0, 1, etc.)
            
        Returns:
            True if loaded successfully
        """
        # Close previous video if open
        if self.cap is not None:
            self.cap.release()
        
        # Open video
        if isinstance(video_path, int) or video_path.isdigit():
            # Camera stream
            self.cap = cv2.VideoCapture(int(video_path))
        else:
            # Video file
            self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.total_frames > 0 and self.fps > 0:
            self.duration_seconds = self.total_frames / self.fps
        
        return True
    
    def extract_keyframes(
        self,
        num_frames: int = 10,
        method: str = 'uniform'  # 'uniform', 'scene_change', 'histogram'
    ) -> List[VideoFrame]:
        """
        Extract keyframes from video
        
        Args:
            num_frames: Number of frames to extract
            method: Extraction method
            
        Returns:
            List of VideoFrame objects
        """
        if self.cap is None:
            raise ValueError("No video loaded")
        
        if method == 'uniform':
            return self._extract_uniform_frames(num_frames)
        elif method == 'scene_change':
            return self._extract_scene_change_frames(num_frames)
        elif method == 'histogram':
            return self._extract_histogram_diff_frames(num_frames)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _extract_uniform_frames(self, num_frames: int) -> List[VideoFrame]:
        """Extract uniformly spaced frames"""
        frames = []
        
        if self.total_frames == 0:
            return frames
        
        # Calculate frame indices to extract
        step = max(1, self.total_frames // num_frames)
        
        for i in range(0, self.total_frames, step):
            if len(frames) >= num_frames:
                break
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            
            if ret:
                timestamp = self._frame_to_timestamp(i)
                frames.append(VideoFrame(
                    frame_number=i,
                    timestamp=timestamp,
                    frame=frame,
                    is_keyframe=True
                ))
        
        return frames
    
    def _extract_scene_change_frames(self, max_frames: int) -> List[VideoFrame]:
        """Extract frames at scene changes"""
        scenes = self.detect_scene_changes(threshold=30.0)
        
        # Take top scene changes
        scenes = sorted(scenes, key=lambda x: x.change_score, reverse=True)[:max_frames]
        scenes = sorted(scenes, key=lambda x: x.frame_number)  # Re-sort by time
        
        frames = []
        for scene in scenes:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, scene.frame_number)
            ret, frame = self.cap.read()
            
            if ret:
                frames.append(VideoFrame(
                    frame_number=scene.frame_number,
                    timestamp=scene.timestamp,
                    frame=frame,
                    is_keyframe=True
                ))
        
        return frames
    
    def _extract_histogram_diff_frames(self, num_frames: int) -> List[VideoFrame]:
        """Extract frames with highest histogram differences"""
        # Sample frames and compute histogram differences
        sample_step = max(1, self.total_frames // (num_frames * 3))
        
        frame_diffs = []
        prev_hist = None
        
        for i in range(0, self.total_frames, sample_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            
            if not ret:
                continue
            
            # Compute histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Compute difference
                diff = np.sum(np.abs(hist - prev_hist))
                frame_diffs.append((i, diff))
            
            prev_hist = hist
        
        # Sort by difference and take top frames
        frame_diffs = sorted(frame_diffs, key=lambda x: x[1], reverse=True)[:num_frames]
        frame_indices = sorted([f[0] for f in frame_diffs])
        
        # Extract frames
        frames = []
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            
            if ret:
                timestamp = self._frame_to_timestamp(idx)
                frames.append(VideoFrame(
                    frame_number=idx,
                    timestamp=timestamp,
                    frame=frame,
                    is_keyframe=True
                ))
        
        return frames
    
    def detect_scene_changes(self, threshold: float = 30.0) -> List[SceneChange]:
        """
        Detect scene changes in video
        
        Args:
            threshold: Histogram difference threshold
            
        Returns:
            List of SceneChange objects
        """
        scenes = []
        prev_hist = None
        
        # Sample every Nth frame for efficiency
        sample_step = max(1, int(self.fps))  # 1 frame per second
        
        for i in range(0, self.total_frames, sample_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                diff = np.sum(np.abs(hist - prev_hist))
                
                if diff > threshold:
                    timestamp = self._frame_to_timestamp(i)
                    scenes.append(SceneChange(
                        frame_number=i,
                        timestamp=timestamp,
                        change_score=float(diff)
                    ))
            
            prev_hist = hist
        
        return scenes
    
    def get_frame_at_timestamp(self, timestamp_str: str) -> Optional[np.ndarray]:
        """
        Get frame at specific timestamp
        
        Args:
            timestamp_str: Timestamp in HH:MM:SS or MM:SS format
            
        Returns:
            Frame as numpy array or None
        """
        # Parse timestamp
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        else:
            return None
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
        frame_number = int(total_seconds * self.fps)
        
        if frame_number >= self.total_frames:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def _frame_to_timestamp(self, frame_number: int) -> str:
        """Convert frame number to timestamp string"""
        if self.fps == 0:
            return "00:00:00"
        
        seconds = frame_number / self.fps
        td = timedelta(seconds=seconds)
        
        # Format as HH:MM:SS
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_video_info(self) -> Dict:
        """Get video information"""
        if self.cap is None:
            return {}
        
        return {
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration_seconds': self.duration_seconds,
            'duration_str': str(timedelta(seconds=int(self.duration_seconds))),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def __del__(self):
        """Release video capture on cleanup"""
        if self.cap is not None:
            self.cap.release()


# Example usage
if __name__ == "__main__":
    processor = VideoProcessor()
    
    # Load video
    if processor.load_video("sample.mp4"):
        # Get info
        info = processor.get_video_info()
        print(f"Video: {info['duration_str']}, {info['fps']} fps")
        
        # Extract keyframes
        keyframes = processor.extract_keyframes(num_frames=5, method='uniform')
        print(f"\nExtracted {len(keyframes)} keyframes:")
        for kf in keyframes:
            print(f"  - Frame {kf.frame_number} @ {kf.timestamp}")
        
        # Detect scenes
        scenes = processor.detect_scene_changes()
        print(f"\nDetected {len(scenes)} scene changes")
