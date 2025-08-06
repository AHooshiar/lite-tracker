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


class CustomMeshTracker:
    def __init__(self, weights_path, device='auto'):
        """
        Custom mesh tracker where users can select points or use a 25x25 grid.
        
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
        
        # Tracking parameters
        self.queries = None
        self.initialized = False
        self.clicked_points = []
        self.window_name = "Click points to track (Press 's' when done, 'c' to clear, 'g' for 25x25 grid)"
        
        # Mesh parameters
        self.triangulation = None
        self.original_points = None
        self.mesh_vertices = None
        self.mesh_faces = None
        self.use_custom_points = False
        
        # Elongation tracking
        self.initial_edge_lengths = None
        self.current_edge_lengths = None
        
        # Performance tracking
        self.processing_times = []
        
        # Toggle states
        self.show_mesh = True
        self.show_lengths = True
        self.paused = False
        
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
            list: List of [x, y] coordinates of selected points, or None for grid
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
        cv2.putText(self.current_frame, "Press 'g' for 25x25 grid", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(self.window_name, self.current_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Start tracking with custom points
                if len(self.clicked_points) > 0:
                    print(f"Starting tracking with {len(self.clicked_points)} custom points")
                    self.use_custom_points = True
                    break
                else:
                    print("Please select at least one point or press 'g' for grid")
                    
            elif key == ord('g'):  # Use 25x25 grid
                print("Using 25x25 grid (625 points)")
                self.use_custom_points = False
                break
                
            elif key == ord('c'):  # Clear points
                self.clicked_points = []
                self.current_frame = frame.copy()
                cv2.putText(self.current_frame, "Click on points to track", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(self.current_frame, "Press 's' when done, 'c' to clear", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(self.current_frame, "Press 'g' for 25x25 grid", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(self.window_name, self.current_frame)
                print("Cleared all points")
                
            elif key == ord('q'):  # Quit
                return None
        
        cv2.destroyWindow(self.window_name)
        return self.clicked_points if self.use_custom_points else None
    
    def initialize_tracking(self, frame, custom_points=None):
        """Initialize tracking with custom points or 25x25 grid."""
        H, W = frame.shape[:2]
        
        if custom_points is not None and len(custom_points) > 0:
            # Use custom points
            points = np.array(custom_points, dtype=np.float32)
            print(f"Using {len(points)} custom points")
        else:
            # Use 25x25 grid
            grid_pts = get_points_on_a_grid(25, (H, W))
            points = grid_pts[0].cpu().numpy()
            print(f"Using 25x25 grid ({len(points)} points)")
        
        # Convert to tensor format for tracking
        points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Create queries: [frame_idx, x, y] format
        frame_indices = torch.zeros(len(points), 1, device=self.device)
        self.queries = torch.cat([frame_indices, points_tensor], dim=1)
        self.queries = self.queries.unsqueeze(0)  # Add batch dimension
        
        # Store original points for mesh generation
        self.original_points = points
        
        # Create optimized mesh with minimum triangles
        self.create_optimized_mesh()
        
        # Initialize edge length tracking
        self.initialize_edge_tracking()
        
        print(f"Initialized tracking with {len(points)} points")
        print(f"Created optimized mesh with {len(self.mesh_faces)} triangles")
        
    def create_optimized_mesh(self):
        """Create mesh with minimum number of triangles."""
        points = self.original_points
        
        if len(points) < 3:
            print("Warning: Need at least 3 points for mesh")
            self.mesh_vertices = points
            self.mesh_faces = np.array([])
            return
        
        # Create triangulation
        try:
            tri = Delaunay(points)
            self.triangulation = tri
            self.mesh_vertices = points
            self.mesh_faces = tri.simplices
            
            # Optimize triangles (remove unnecessary ones)
            self.optimize_triangles()
            
        except Exception as e:
            print(f"Warning: Could not create triangulation: {e}")
            # Fallback to simple mesh
            self.create_simple_mesh(points)
    
    def optimize_triangles(self):
        """Optimize triangles to use minimum number."""
        if len(self.mesh_faces) == 0:
            return
            
        # Remove triangles with very small area
        min_area = 10.0  # Minimum triangle area
        valid_faces = []
        
        for face in self.mesh_faces:
            # Calculate triangle area
            v1, v2, v3 = self.mesh_vertices[face]
            area = abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])) / 2
            if area > min_area:
                valid_faces.append(face)
        
        self.mesh_faces = np.array(valid_faces)
        print(f"Optimized mesh: {len(self.mesh_faces)} triangles (removed small triangles)")
    
    def create_simple_mesh(self, points):
        """Create a simple mesh if triangulation fails."""
        if len(points) < 3:
            self.mesh_vertices = points
            self.mesh_faces = np.array([])
            return
            
        # Create minimal triangulation
        faces = []
        for i in range(len(points) - 2):
            faces.append([i, i+1, i+2])
        
        self.mesh_vertices = points
        self.mesh_faces = np.array(faces)
    
    def initialize_edge_tracking(self):
        """Initialize edge length tracking for elongation measurement."""
        if self.mesh_faces is None or len(self.mesh_faces) == 0:
            self.initial_edge_lengths = {}
            self.current_edge_lengths = {}
            return
        
        self.initial_edge_lengths = {}
        self.current_edge_lengths = {}
        
        # Calculate initial edge lengths for each triangle
        for face_idx, face in enumerate(self.mesh_faces):
            for i in range(3):
                v1_idx = face[i]
                v2_idx = face[(i + 1) % 3]
                
                # Create unique edge key (smaller index first)
                edge_key = tuple(sorted([v1_idx, v2_idx]))
                
                if edge_key not in self.initial_edge_lengths:
                    v1 = self.mesh_vertices[v1_idx]
                    v2 = self.mesh_vertices[v2_idx]
                    length = np.linalg.norm(v2 - v1)
                    self.initial_edge_lengths[edge_key] = length
                    self.current_edge_lengths[edge_key] = length
        
        print(f"Initialized tracking for {len(self.initial_edge_lengths)} edges")
    
    def update_edge_lengths(self, coords_np):
        """Update current edge lengths based on tracked coordinates."""
        if self.initial_edge_lengths is None:
            return
        
        self.current_edge_lengths = {}
        
        # Calculate current edge lengths
        for face_idx, face in enumerate(self.mesh_faces):
            for i in range(3):
                v1_idx = face[i]
                v2_idx = face[(i + 1) % 3]
                
                # Create unique edge key
                edge_key = tuple(sorted([v1_idx, v2_idx]))
                
                if edge_key in self.initial_edge_lengths:
                    v1 = coords_np[v1_idx]
                    v2 = coords_np[v2_idx]
                    length = np.linalg.norm(v2 - v1)
                    self.current_edge_lengths[edge_key] = length
    
    def get_edge_elongation(self, edge_key):
        """Calculate elongation ratio for an edge."""
        if (edge_key in self.initial_edge_lengths and 
            edge_key in self.current_edge_lengths):
            initial_length = self.initial_edge_lengths[edge_key]
            current_length = self.current_edge_lengths[edge_key]
            
            if initial_length > 0:
                return current_length / initial_length
        return 1.0  # No elongation
    
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
        Draw mesh with toggleable features and improved visualization.
        
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
        
        # Update edge lengths
        self.update_edge_lengths(coords_np)
        
        # Create a copy of the frame for drawing
        frame_with_mesh = frame.copy()
        
        # Draw mesh edges (toggleable)
        if self.mesh_faces is not None and len(self.mesh_faces) > 0 and self.show_mesh:
            drawn_edges = set()  # Track drawn edges to avoid duplicates
            
            for face in self.mesh_faces:
                # Draw each edge of the triangle
                for i in range(3):
                    v1_idx = face[i]
                    v2_idx = face[(i + 1) % 3]
                    
                    # Create unique edge key
                    edge_key = tuple(sorted([v1_idx, v2_idx]))
                    
                    # Only draw each edge once
                    if edge_key not in drawn_edges:
                        drawn_edges.add(edge_key)
                        
                        # Check if both vertices are visible
                        if (v1_idx < len(coords_np) and v2_idx < len(coords_np) and
                            visibility_np[v1_idx] and visibility_np[v2_idx]):
                            
                            # Get vertex coordinates
                            x1, y1 = int(coords_np[v1_idx][0]), int(coords_np[v1_idx][1])
                            x2, y2 = int(coords_np[v2_idx][0]), int(coords_np[v2_idx][1])
                            
                            # Calculate elongation
                            elongation = self.get_edge_elongation(edge_key)
                            
                            # Color based on elongation
                            # Blue = compression, Red = expansion, Green = no change
                            if elongation < 1.0:  # Compression
                                # Blue color (BGR format)
                                color = (255, 0, 0)  # Blue
                                intensity = int(255 * (1.0 - elongation))
                            elif elongation > 1.0:  # Expansion
                                # Red color
                                color = (0, 0, 255)  # Red
                                intensity = int(255 * (elongation - 1.0))
                            else:  # No change
                                color = (0, 255, 0)  # Green
                                intensity = 0
                            
                            # Clamp intensity
                            intensity = min(255, max(0, intensity))
                            
                            # Create color with transparency
                            alpha = 0.5  # 50% transparency
                            overlay_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                            
                            # Draw edge with half thickness
                            thickness = max(1, int(1 + intensity / 100))  # Half the original thickness
                            cv2.line(frame_with_mesh, (x1, y1), (x2, y2), overlay_color, thickness)
                            
                            # Draw elongation value on edge (toggleable)
                            if elongation != 1.0 and self.show_lengths:
                                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                                cv2.putText(frame_with_mesh, f"{elongation:.2f}", (mid_x, mid_y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, overlay_color, 1)
        
        # Draw mesh vertices (tracked points) - hollow circles
        for i in range(len(coords_np)):
            if visibility_np[i]:
                x, y = int(coords_np[i][0]), int(coords_np[i][1])
                
                # Draw hollow circle (white outline)
                cv2.circle(frame_with_mesh, (x, y), 4, (255, 255, 255), 2)  # Hollow circle
                
                # Draw vertex number for first few points
                if i < 10:
                    cv2.putText(frame_with_mesh, str(i), (x+6, y-6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw performance info
        fps = self.get_average_fps()
        cv2.putText(frame_with_mesh, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_mesh, f"Vertices: {len(coords_np)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_mesh, f"Edges: {len(self.initial_edge_lengths)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show mode
        mode = "Custom Points" if self.use_custom_points else "25x25 Grid"
        cv2.putText(frame_with_mesh, f"Mode: {mode}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show toggle states
        cv2.putText(frame_with_mesh, f"Mesh: {'ON' if self.show_mesh else 'OFF'}", (10, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_mesh, f"Lengths: {'ON' if self.show_lengths else 'OFF'}", (10, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_mesh, f"Paused: {'YES' if self.paused else 'NO'}", (10, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add controls legend
        cv2.putText(frame_with_mesh, "Controls: M=Mesh, L=Lengths, SPACE=Pause, Q=Quit", (10, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame_with_mesh
    
    def run_webcam(self, camera_id=0):
        """Run custom mesh tracking on webcam."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Get first frame for point selection
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Let user select points or choose grid
        custom_points = self.select_points_interactively(frame)
        
        # Initialize tracking
        self.initialize_tracking(frame, custom_points)
        
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
            cv2.imshow('Custom Mesh Tracker', frame)
            
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
                    custom_points = self.select_points_interactively(frame)
                    self.initialize_tracking(frame, custom_points)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video_file(self, video_path):
        """Run custom mesh tracking on a video file and save the result."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get first frame for point selection
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Let user select points or choose grid
        custom_points = self.select_points_interactively(frame)
        
        # Initialize tracking
        self.initialize_tracking(frame, custom_points)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        
        # Prepare video writer
        output_filename = self.prepare_video_writer(video_path, fps, frame.shape)
        
        print(f"Video FPS: {fps}")
        print(f"Saving output to: {output_filename}")
        print("Controls: M=Mesh, L=Lengths, SPACE=Pause, Q=Quit, R=Reset")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            # Process frame
            coords, visibility, confidence, proc_time = self.process_frame(frame)
            
            # Draw mesh
            frame_with_mesh = self.draw_mesh_on_frame(frame, coords, visibility, confidence)
            
            # Save frame to video
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.write(frame_with_mesh)
            
            # Display
            cv2.imshow('Custom Mesh Tracker', frame_with_mesh)
            
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
                
                # Close current video writer and start new one
                if hasattr(self, 'video_writer') and self.video_writer is not None:
                    self.video_writer.release()
                
                # Go back to first frame and select new points
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    custom_points = self.select_points_interactively(frame)
                    self.initialize_tracking(frame, custom_points)
                    output_filename = self.prepare_video_writer(video_path, fps, frame.shape)
                    print(f"New output file: {output_filename}")
            elif key == ord('m'):
                self.show_mesh = not self.show_mesh
                print(f"Mesh display: {'ON' if self.show_mesh else 'OFF'}")
            elif key == ord('l'):
                self.show_lengths = not self.show_lengths
                print(f"Length display: {'ON' if self.show_lengths else 'OFF'}")
            elif key == 32:  # Spacebar
                self.paused = not self.paused
                print(f"Playback: {'PAUSED' if self.paused else 'PLAYING'}")
                if self.paused:
                    # Wait for another key press to resume
                    cv2.waitKey(0)
            
            frame_count += 1
        
        # Clean up
        cap.release()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"Video saved successfully: {output_filename}")
    
    def prepare_video_writer(self, video_path, fps, frame_shape):
        """Prepare video writer for saving the output."""
        # Create results directory if it doesn't exist
        import os
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate output filename
        input_filename = Path(video_path).stem
        if self.use_custom_points:
            mode_suffix = f"custom_{len(self.original_points)}pts"
        else:
            mode_suffix = "grid"
        output_filename = f"{results_dir}/{input_filename}_mesh_{mode_suffix}.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_filename, 
            fourcc, 
            fps, 
            (frame_shape[1], frame_shape[0])
        )
        
        return output_filename


def main():
    parser = argparse.ArgumentParser(description='Custom mesh tracking with LiteTracker')
    parser.add_argument('-w', '--weights', required=True, help='Path to model weights')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('-v', '--video', help='Video file path (optional, uses webcam if not provided)')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'], 
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Set OpenMP environment variable first
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Create tracker
    tracker = CustomMeshTracker(
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