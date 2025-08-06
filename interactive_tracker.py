import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import time

from src.lite_tracker import LiteTracker


class InteractiveTracker:
    def __init__(self, weights_path, device='auto'):
        """
        Interactive tracker where users click on points of interest.
        
        Args:
            weights_path (str): Path to model weights
            device (str): Device to run on
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
        
        # Tracking state
        self.queries = None
        self.initialized = False
        self.clicked_points = []
        self.window_name = "Click points of interest (Press 's' when done, 'c' to clear)"
        
        # Performance tracking
        self.processing_times = []
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select points of interest."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append([x, y])
            print(f"Added point {len(self.clicked_points)}: ({x}, {y})")
            
            # Draw the point on the image
            cv2.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.current_frame, str(len(self.clicked_points)), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.current_frame)
    
    def select_points_interactively(self, frame):
        """
        Let user click on points of interest in the frame.
        
        Args:
            frame (np.ndarray): First frame for point selection
            
        Returns:
            list: List of [x, y] coordinates of selected points
        """
        self.current_frame = frame.copy()
        self.clicked_points = []
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display instructions
        cv2.putText(self.current_frame, "Click on points to track", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.current_frame, "Press 's' when done, 'c' to clear", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(self.window_name, self.current_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Start tracking
                if len(self.clicked_points) > 0:
                    print(f"Starting tracking with {len(self.clicked_points)} points")
                    break
                else:
                    print("Please select at least one point")
                    
            elif key == ord('c'):  # Clear points
                self.clicked_points = []
                self.current_frame = frame.copy()
                cv2.putText(self.current_frame, "Click on points to track", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(self.current_frame, "Press 's' when done, 'c' to clear", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(self.window_name, self.current_frame)
                print("Cleared all points")
                
            elif key == ord('q'):  # Quit
                return None
        
        cv2.destroyWindow(self.window_name)
        return self.clicked_points
    
    def initialize_tracking(self, points):
        """
        Initialize tracking with user-selected points.
        
        Args:
            points (list): List of [x, y] coordinates
        """
        if not points:
            raise ValueError("No points provided")
            
        # Convert points to tensor format
        points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Create queries: [frame_idx, x, y] format
        frame_indices = torch.zeros(len(points), 1, device=self.device)
        self.queries = torch.cat([frame_indices, points_tensor], dim=1)
        self.queries = self.queries.unsqueeze(0)  # Add batch dimension
        
        self.initialized = True
        print(f"Initialized tracking with {len(points)} points")
        
    def track(self, frame):
        """
        Track the selected points in the given frame.
        
        Args:
            frame (np.ndarray): Input frame (H, W, C)
            
        Returns:
            tuple: (coords, visibility, confidence, processing_time)
        """
        if not self.initialized:
            raise ValueError("Call initialize_tracking() first")
            
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
        if len(self.processing_times) > 30:
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
                cv2.circle(frame, (x, y), 5, color, -1)
                
                # Draw point number
                cv2.putText(frame, str(i+1), (x+8, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw confidence
                cv2.putText(frame, f"{conf:.2f}", (x+8, y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw performance info
        fps = self.get_average_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Points: {len(coords_np)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def run_webcam(self, camera_id=0):
        """Run interactive tracking on webcam."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Get first frame for point selection
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Let user select points
        points = self.select_points_interactively(frame)
        if points is None:
            print("No points selected, exiting")
            cap.release()
            return
        
        # Initialize tracking
        self.initialize_tracking(points)
        
        print("Press 'q' to quit, 'r' to reset tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            coords, visibility, confidence, proc_time = self.track(frame)
            
            # Draw results
            frame = self.draw_tracks(frame, coords, visibility, confidence)
            
            # Display
            cv2.imshow('Interactive LiteTracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting tracking...")
                self.model.reset()
                self.queries = None
                self.initialized = False
                self.processing_times.clear()
                
                # Get new frame and select new points
                ret, frame = cap.read()
                if ret:
                    points = self.select_points_interactively(frame)
                    if points:
                        self.initialize_tracking(points)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video_file(self, video_path):
        """Run interactive tracking on a video file."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get first frame for point selection
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Let user select points
        points = self.select_points_interactively(frame)
        if points is None:
            print("No points selected, exiting")
            cap.release()
            return
        
        # Initialize tracking
        self.initialize_tracking(points)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        
        print(f"Video FPS: {fps}")
        print("Press 'q' to quit, 'r' to reset tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            # Process frame
            coords, visibility, confidence, proc_time = self.track(frame)
            
            # Draw results
            frame = self.draw_tracks(frame, coords, visibility, confidence)
            
            # Display
            cv2.imshow('Interactive LiteTracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting tracking...")
                self.model.reset()
                self.queries = None
                self.initialized = False
                self.processing_times.clear()
                
                # Go back to first frame and select new points
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    points = self.select_points_interactively(frame)
                    if points:
                        self.initialize_tracking(points)
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Set OpenMP environment variable first
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    parser = argparse.ArgumentParser(description='Interactive tracking with LiteTracker')
    parser.add_argument('-w', '--weights', required=True, help='Path to model weights')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('-v', '--video', help='Video file path (optional, uses webcam if not provided)')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'], 
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Create tracker
    tracker = InteractiveTracker(
        weights_path=args.weights,
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