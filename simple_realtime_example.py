import cv2
import torch
import numpy as np
import time

from src.lite_tracker import LiteTracker
from src.model_utils import get_points_on_a_grid


class SimpleRealtimeTracker:
    """
    Simple real-time tracker for integration into existing applications.
    
    Usage:
        tracker = SimpleRealtimeTracker('weights/scaled_online.pth')
        tracker.initialize(frame)  # Call once with first frame
        coords, visibility, confidence = tracker.track(frame)  # Call for each frame
    """
    
    def __init__(self, weights_path, grid_size=10, device='auto'):
        # Set device
        if device == 'auto':
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = device
            
        # Load model
        self.model = LiteTracker()
        with open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.grid_size = grid_size
        self.queries = None
        self.initialized = False
        
    def initialize(self, frame):
        """Initialize tracking on the first frame."""
        H, W = frame.shape[:2]
        grid_pts = get_points_on_a_grid(self.grid_size, (H, W))
        self.queries = torch.cat([
            torch.ones_like(grid_pts[:, :, :1]) * 0,
            grid_pts,
        ], dim=2).to(self.device)
        self.initialized = True
        
    def track(self, frame):
        """
        Track points in the given frame.
        
        Args:
            frame (np.ndarray): Input frame (H, W, C)
            
        Returns:
            tuple: (coords, visibility, confidence)
                - coords: numpy array of shape (N, 2) with (x, y) coordinates
                - visibility: numpy array of shape (N,) with boolean visibility
                - confidence: numpy array of shape (N,) with confidence scores
        """
        if not self.initialized:
            raise ValueError("Call initialize() first with the first frame")
            
        # Convert frame to tensor
        frame_tensor = (
            torch.tensor(frame, device=self.device)
            .permute(2, 0, 1)[None]
            .float()
        )
        
        # Run tracking
        with torch.no_grad():
            coords, visibility, confidence = self.model(frame_tensor, self.queries)
        
        # Convert to numpy
        coords_np = coords[0, 0].cpu().numpy()  # (N, 2)
        visibility_np = visibility[0, 0].cpu().numpy()  # (N,)
        confidence_np = confidence[0, 0].cpu().numpy()  # (N,)
        
        return coords_np, visibility_np, confidence_np
    
    def reset(self):
        """Reset the tracker for a new sequence."""
        self.model.reset()
        self.queries = None
        self.initialized = False


# Example usage
def example_usage():
    """Example of how to use the simple tracker."""
    
    # Initialize tracker
    tracker = SimpleRealtimeTracker('weights/scaled_online.pth', grid_size=10)
    
    # Open video capture (webcam or video file)
    cap = cv2.VideoCapture(0)  # or cv2.VideoCapture('video.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Initialize on first frame
    ret, frame = cap.read()
    if ret:
        tracker.initialize(frame)
        print(f"Initialized tracking with {tracker.grid_size**2} points")
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Track points
        start_time = time.time()
        coords, visibility, confidence = tracker.track(frame)
        processing_time = time.time() - start_time
        
        # Draw results
        for i in range(len(coords)):
            if visibility[i]:
                x, y = int(coords[i][0]), int(coords[i][1])
                conf = confidence[i]
                
                # Color based on confidence
                color = (0, int(255 * conf), int(255 * (1 - conf)))
                cv2.circle(frame, (x, y), 3, color, -1)
        
        # Draw performance info
        fps = 1.0 / processing_time if processing_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Tracking', frame)
        
        # Handle key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Set OpenMP environment variable
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    example_usage() 