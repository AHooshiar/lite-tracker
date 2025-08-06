import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import time

from src.model_utils import get_points_on_a_grid
from src.lite_tracker import LiteTracker


class RealtimeTracker:
    def __init__(self, weights_path, grid_size=10, device='auto'):
        """
        Initialize real-time tracker with LiteTracker.
        
        Args:
            weights_path (str): Path to the model weights
            grid_size (int): Number of tracking points (grid_size x grid_size)
            device (str): Device to run on ('cpu', 'cuda', 'mps', or 'auto')
        """
        # Set device
        if device == 'auto':
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = LiteTracker()
        with open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize tracking parameters
        self.grid_size = grid_size
        self.queries = None
        self.frame_count = 0
        
        # Performance tracking
        self.fps_history = []
        self.processing_times = []
        
    def initialize_tracking(self, frame):
        """Initialize tracking grid on the first frame."""
        H, W = frame.shape[:2]
        grid_pts = get_points_on_a_grid(self.grid_size, (H, W))
        self.queries = torch.cat([
            torch.ones_like(grid_pts[:, :, :1]) * 0,  # Start from frame 0
            grid_pts,
        ], dim=2).to(self.device)
        
        print(f"Initialized {self.grid_size}x{self.grid_size} = {self.grid_size**2} tracking points")
        
    def process_frame(self, frame):
        """
        Process a single frame and return tracking results.
        
        Args:
            frame (np.ndarray): Input frame (H, W, C)
            
        Returns:
            tuple: (coords, visibility, confidence, processing_time)
        """
        start_time = time.time()
        
        # Convert frame to tensor
        frame_tensor = (
            torch.tensor(frame, device=self.device)
            .permute(2, 0, 1)[None]
            .float()
        )
        
        # Run tracking
        with torch.no_grad():
            coords, visibility, confidence = self.model(frame_tensor, self.queries)
        
        processing_time = time.time() - start_time
        
        # Update performance metrics
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:  # Keep last 30 frames
            self.processing_times.pop(0)
        
        return coords, visibility, confidence, processing_time
    
    def get_average_fps(self):
        """Get average FPS from recent processing times."""
        if not self.processing_times:
            return 0
        avg_time = np.mean(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def draw_tracks(self, frame, coords, visibility, confidence):
        """
        Draw tracking results on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            coords (torch.Tensor): Track coordinates
            visibility (torch.Tensor): Visibility mask
            confidence (torch.Tensor): Confidence scores
            
        Returns:
            np.ndarray: Frame with tracks drawn
        """
        # Convert tensors to numpy
        coords_np = coords[0, 0].cpu().numpy()  # Shape: (N, 2)
        visibility_np = visibility[0, 0].cpu().numpy()  # Shape: (N,)
        confidence_np = confidence[0, 0].cpu().numpy()  # Shape: (N,)
        
        # Draw each tracked point
        for i in range(len(coords_np)):
            if visibility_np[i]:
                x, y = int(coords_np[i][0]), int(coords_np[i][1])
                conf = confidence_np[i]
                
                # Color based on confidence (green=high, red=low)
                color = (0, int(255 * conf), int(255 * (1 - conf)))
                
                # Draw circle
                cv2.circle(frame, (x, y), 3, color, -1)
                
                # Draw confidence text
                cv2.putText(frame, f"{conf:.2f}", (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw performance info
        fps = self.get_average_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {self.grid_size**2}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def run_webcam(self, camera_id=0):
        """Run real-time tracking on webcam."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("Press 'q' to quit, 'r' to reset tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Initialize tracking on first frame
            if self.queries is None:
                self.initialize_tracking(frame)
            
            # Process frame
            coords, visibility, confidence, proc_time = self.process_frame(frame)
            
            # Draw results
            frame = self.draw_tracks(frame, coords, visibility, confidence)
            
            # Display
            cv2.imshow('LiteTracker - Real-time Tracking', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting tracking...")
                self.model.reset()
                self.queries = None
                self.frame_count = 0
                self.processing_times.clear()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video_file(self, video_path):
        """Run tracking on a video file."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        
        print(f"Video FPS: {fps}")
        print("Press 'q' to quit, 'r' to reset tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            # Initialize tracking on first frame
            if self.queries is None:
                self.initialize_tracking(frame)
            
            # Process frame
            coords, visibility, confidence, proc_time = self.process_frame(frame)
            
            # Draw results
            frame = self.draw_tracks(frame, coords, visibility, confidence)
            
            # Display
            cv2.imshow('LiteTracker - Video Tracking', frame)
            
            # Handle key presses
            key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting tracking...")
                self.model.reset()
                self.queries = None
                self.frame_count = 0
                self.processing_times.clear()
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Real-time tracking with LiteTracker')
    parser.add_argument('-w', '--weights', required=True, help='Path to model weights')
    parser.add_argument('-s', '--grid_size', type=int, default=10, help='Grid size for tracking points')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('-v', '--video', help='Video file path (optional, uses webcam if not provided)')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'], 
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Set OpenMP environment variable
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Create tracker
    tracker = RealtimeTracker(
        weights_path=args.weights,
        grid_size=args.grid_size,
        device=args.device
    )
    
    # Run tracking
    if args.video:
        print(f"Running on video: {args.video}")
        tracker.run_video_file(args.video)
    else:
        print(f"Running on webcam (camera {args.camera})")
        tracker.run_webcam(args.camera)


if __name__ == "__main__":
    main() 