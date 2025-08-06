import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import time
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from src.lite_tracker import LiteTracker
from src.model_utils import get_points_on_a_grid


class MeshTracker:
    def __init__(self, weights_path, grid_size=10, device='auto'):
        """
        Mesh-based tracker that creates a deformable mesh from tracked points.
        
        Args:
            weights_path (str): Path to model weights
            grid_size (int): Grid size for tracking points
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
        
        # Tracking parameters
        self.grid_size = grid_size
        self.queries = None
        self.initialized = False
        
        # Mesh parameters
        self.triangulation = None
        self.original_points = None
        self.mesh_vertices = None
        self.mesh_faces = None
        
        # Performance tracking
        self.processing_times = []
        
    def initialize_tracking(self, frame):
        """Initialize tracking grid on the first frame."""
        H, W = frame.shape[:2]
        grid_pts = get_points_on_a_grid(self.grid_size, (H, W))
        self.queries = torch.cat([
            torch.ones_like(grid_pts[:, :, :1]) * 0,  # Start from frame 0
            grid_pts,
        ], dim=2).to(self.device)
        
        # Store original points for mesh generation
        self.original_points = grid_pts[0].cpu().numpy()
        
        # Create triangulation
        self.create_mesh()
        
        print(f"Initialized tracking with {self.grid_size}x{self.grid_size} = {self.grid_size**2} points")
        print(f"Created mesh with {len(self.mesh_faces)} triangles")
        
    def create_mesh(self):
        """Create triangular mesh from grid points."""
        points = self.original_points
        
        # Create triangulation
        try:
            tri = Delaunay(points)
            self.triangulation = tri
            self.mesh_vertices = points
            self.mesh_faces = tri.simplices
        except Exception as e:
            print(f"Warning: Could not create triangulation: {e}")
            # Fallback to simple grid-based mesh
            self.create_grid_mesh(points)
    
    def create_grid_mesh(self, points):
        """Create a simple grid-based mesh if Delaunay fails."""
        # Reshape points to grid
        grid_points = points.reshape(self.grid_size, self.grid_size, 2)
        
        faces = []
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size - 1):
                # Create two triangles for each grid cell
                idx = i * self.grid_size + j
                faces.append([idx, idx + 1, idx + self.grid_size])
                faces.append([idx + 1, idx + self.grid_size + 1, idx + self.grid_size])
        
        self.mesh_vertices = points
        self.mesh_faces = np.array(faces)
    
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
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        return coords, visibility, confidence, processing_time
    
    def get_average_fps(self):
        """Get average FPS from recent processing times."""
        if not self.processing_times:
            return 0
        avg_time = np.mean(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def draw_mesh_on_frame(self, frame, coords, visibility, confidence):
        """
        Draw the deformable mesh on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            coords (torch.Tensor): Track coordinates
            visibility (torch.Tensor): Visibility mask
            confidence (torch.Tensor): Confidence scores
            
        Returns:
            np.ndarray: Frame with mesh drawn
        """
        # Convert tensors to numpy
        coords_np = coords[0, 0].cpu().numpy()  # Shape: (N, 2)
        visibility_np = visibility[0, 0].cpu().numpy()  # Shape: (N,)
        confidence_np = confidence[0, 0].cpu().numpy()  # Shape: (N,)
        
        # Create a copy of the frame for drawing
        frame_with_mesh = frame.copy()
        
        # Draw mesh triangles
        if self.mesh_faces is not None:
            for face in self.mesh_faces:
                # Get vertices of this triangle
                vertices = []
                visible_vertices = 0
                
                for vertex_idx in face:
                    if vertex_idx < len(coords_np) and visibility_np[vertex_idx]:
                        x, y = int(coords_np[vertex_idx][0]), int(coords_np[vertex_idx][1])
                        vertices.append((x, y))
                        visible_vertices += 1
                
                # Only draw triangle if at least 2 vertices are visible
                if visible_vertices >= 2:
                    if len(vertices) == 3:
                        # Draw filled triangle
                        pts = np.array(vertices, np.int32)
                        cv2.fillPoly(frame_with_mesh, [pts], (0, 100, 0, 50))
                        cv2.polylines(frame_with_mesh, [pts], True, (0, 255, 0), 1)
                    elif len(vertices) == 2:
                        # Draw line if only 2 vertices visible
                        cv2.line(frame_with_mesh, vertices[0], vertices[1], (0, 255, 0), 1)
        
        # Draw mesh vertices (tracked points)
        for i in range(len(coords_np)):
            if visibility_np[i]:
                x, y = int(coords_np[i][0]), int(coords_np[i][1])
                conf = confidence_np[i]
                
                # Color based on confidence (green=high, red=low)
                color = (0, int(255 * conf), int(255 * (1 - conf)))
                
                # Draw vertex
                cv2.circle(frame_with_mesh, (x, y), 3, color, -1)
                
                # Draw vertex number for first few points
                if i < 10:
                    cv2.putText(frame_with_mesh, str(i), (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw performance info
        fps = self.get_average_fps()
        cv2.putText(frame_with_mesh, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_mesh, f"Vertices: {len(coords_np)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_mesh, f"Triangles: {len(self.mesh_faces)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame_with_mesh
    
    def create_3d_mesh_visualization(self, coords, visibility):
        """
        Create a 3D visualization of the mesh deformation.
        
        Args:
            coords (torch.Tensor): Track coordinates
            visibility (torch.Tensor): Visibility mask
        """
        coords_np = coords[0, 0].cpu().numpy()
        visibility_np = visibility[0, 0].cpu().numpy()
        
        # Filter visible points
        visible_coords = coords_np[visibility_np]
        
        if len(visible_coords) < 3:
            return None
        
        # Create 3D coordinates (add z-coordinate based on confidence or position)
        z_coords = np.zeros(len(visible_coords))
        coords_3d = np.column_stack([visible_coords, z_coords])
        
        # Create triangulation for visible points
        try:
            tri = Delaunay(visible_coords)
            return coords_3d, tri.simplices
        except:
            return None
    
    def run_webcam(self, camera_id=0):
        """Run mesh tracking on webcam."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Get first frame for initialization
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Initialize tracking
        self.initialize_tracking(frame)
        
        print("Press 'q' to quit, 'r' to reset tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            coords, visibility, confidence, proc_time = self.process_frame(frame)
            
            # Draw mesh
            frame = self.draw_mesh_on_frame(frame, coords, visibility, confidence)
            
            # Display
            cv2.imshow('Mesh Tracker - Real-time Deformation', frame)
            
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
                
                # Reinitialize
                ret, frame = cap.read()
                if ret:
                    self.initialize_tracking(frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video_file(self, video_path):
        """Run mesh tracking on a video file."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get first frame for initialization
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Initialize tracking
        self.initialize_tracking(frame)
        
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
            coords, visibility, confidence, proc_time = self.process_frame(frame)
            
            # Draw mesh
            frame = self.draw_mesh_on_frame(frame, coords, visibility, confidence)
            
            # Display
            cv2.imshow('Mesh Tracker - Video Deformation', frame)
            
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
                
                # Go back to first frame and reinitialize
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    self.initialize_tracking(frame)
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Mesh-based tracking with LiteTracker')
    parser.add_argument('-w', '--weights', required=True, help='Path to model weights')
    parser.add_argument('-s', '--grid_size', type=int, default=10, help='Grid size for tracking points')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('-v', '--video', help='Video file path (optional, uses webcam if not provided)')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'], 
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Set OpenMP environment variable first
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Create tracker
    tracker = MeshTracker(
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