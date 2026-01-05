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
        # Set device - prioritize Metal (MPS) first for macOS
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Ensure MPS is properly configured
        if self.device == "mps":
            # Enable MPS fallback for unsupported operations
            import os
            if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            # Verify MPS is available
            if not torch.backends.mps.is_available():
                print("Warning: MPS requested but not available, falling back to CPU")
                self.device = "cpu"
            
        print(f"Using device: {self.device}")
        if self.device == "mps":
            print("Metal Performance Shaders (MPS) enabled - running on Apple GPU")
        
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
        
        # Interactive point selection
        self.selected_points = []
        self.selecting_points = False
        self.window_name = 'Mesh Tracker - Select Points'
        
        # Mesh parameters
        self.triangulation = None
        self.original_points = None
        self.mesh_vertices = None
        self.mesh_faces = None
        
        # Performance tracking
        self.processing_times = []
        
        # Video writer for saving output
        self.video_writer = None
        
    def initialize_tracking(self, frame, custom_points=None):
        """Initialize tracking with custom points or grid on the first frame."""
        H, W = frame.shape[:2]
        
        if custom_points is not None and len(custom_points) > 0:
            # Use custom selected points
            # Convert to numpy array and ensure correct format: (N, 2) with (x, y) coordinates
            points_array = np.array(custom_points, dtype=np.float32)  # Shape: (N, 2)
            # Points are already in pixel coordinates (x, y)
            # Convert to tensor format: (1, N, 2) - same as get_points_on_a_grid returns
            grid_pts = torch.tensor(points_array, device=self.device).unsqueeze(0)  # Shape: (1, N, 2)
            self.original_points = points_array
            print(f"Initialized tracking with {len(custom_points)} custom points")
        else:
            # Use grid points
            grid_pts = get_points_on_a_grid(self.grid_size, (H, W), device=self.device)
            self.original_points = grid_pts[0].cpu().numpy()
            print(f"Initialized tracking with {self.grid_size}x{self.grid_size} = {self.grid_size**2} grid points")
        
        self.queries = torch.cat([
            torch.ones_like(grid_pts[:, :, :1]) * 0,  # Start from frame 0
            grid_pts,
        ], dim=2).to(self.device)
        
        # Create triangulation
        self.create_mesh()
        
        print(f"Created mesh with {len(self.mesh_faces)} triangles")
        
    def create_mesh(self):
        """Create triangular mesh from grid points."""
        points = self.original_points
        
        # Need at least 3 points for triangulation
        if len(points) < 3:
            print(f"Warning: Need at least 3 points for mesh, got {len(points)}")
            self.mesh_vertices = points
            self.mesh_faces = np.array([], dtype=np.int32)
            self.triangulation = None
            return
        
        # Create triangulation
        try:
            tri = Delaunay(points)
            self.triangulation = tri
            self.mesh_vertices = points
            self.mesh_faces = tri.simplices
        except Exception as e:
            print(f"Warning: Could not create triangulation: {e}")
            # Fallback to simple grid-based mesh if it's a grid
            if hasattr(self, 'grid_size') and len(points) == self.grid_size ** 2:
                self.create_grid_mesh(points)
            else:
                # For custom points, just store vertices without faces
                self.mesh_vertices = points
                self.mesh_faces = np.array([], dtype=np.int32)
                self.triangulation = None
    
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
        
        # Create overlay for transparent triangles
        overlay = frame_with_mesh.copy()
        
        # Blue color in BGR format
        blue_color = (255, 0, 0)
        
        # Draw mesh triangles with transparency
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
                        # Draw filled triangle with transparency on overlay
                        pts = np.array(vertices, np.int32)
                        cv2.fillPoly(overlay, [pts], blue_color)
                    elif len(vertices) == 2:
                        # Draw line if only 2 vertices visible (will be drawn as edge below)
                        pass
        
        # Blend overlay with original frame for transparency (alpha = 0.2 for more transparency)
        cv2.addWeighted(overlay, 0.2, frame_with_mesh, 0.8, 0, frame_with_mesh)
        
        # Draw all mesh edges (legs) in blue
        if self.mesh_faces is not None:
            for face in self.mesh_faces:
                # Get vertices of this triangle
                vertices = []
                for vertex_idx in face:
                    if vertex_idx < len(coords_np) and visibility_np[vertex_idx]:
                        x, y = int(coords_np[vertex_idx][0]), int(coords_np[vertex_idx][1])
                        vertices.append((x, y))
                
                # Draw edges connecting vertices
                if len(vertices) == 3:
                    # Draw all three edges of the triangle
                    cv2.line(frame_with_mesh, vertices[0], vertices[1], blue_color, 2)
                    cv2.line(frame_with_mesh, vertices[1], vertices[2], blue_color, 2)
                    cv2.line(frame_with_mesh, vertices[2], vertices[0], blue_color, 2)
                elif len(vertices) == 2:
                    # Draw line if only 2 vertices visible
                    cv2.line(frame_with_mesh, vertices[0], vertices[1], blue_color, 2)
        
        # Draw mesh vertices (tracked points) in blue
        for i in range(len(coords_np)):
            if visibility_np[i]:
                x, y = int(coords_np[i][0]), int(coords_np[i][1])
                
                # Blue color for nodes
                node_color = blue_color
                
                # Draw vertex circle in blue
                cv2.circle(frame_with_mesh, (x, y), 4, node_color, -1)
                cv2.circle(frame_with_mesh, (x, y), 4, (255, 255, 255), 1)  # White outline
                
                # Draw vertex number for first few points
                if i < 10:
                    cv2.putText(frame_with_mesh, str(i), (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
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
        
        # Get first frame for point selection
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Interactive point selection
        if not self.select_points_interactive(frame):
            cap.release()
            cv2.destroyAllWindows()
            return
        
        # Initialize tracking with selected points
        self.initialize_tracking(frame, custom_points=self.selected_points if self.selected_points else None)
        
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
                    # Re-select points
                    if not self.select_points_interactive(frame):
                        break
                    self.initialize_tracking(frame, custom_points=self.selected_points if self.selected_points else None)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN and self.selecting_points:
            self.selected_points.append([x, y])
            print(f"Selected point {len(self.selected_points)}: ({x}, {y})")
    
    def draw_instructions(self, frame):
        """Draw instructions on the frame."""
        instructions = [
            "INSTRUCTIONS:",
            "1. Click on points you want to track",
            "2. Press SPACEBAR to start tracking",
            "3. Press 'c' to clear all points",
            "4. Press 'q' to quit"
        ]
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw instructions
        y_offset = 35
        for i, instruction in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            font_scale = 0.7 if i == 0 else 0.6
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, instruction, (20, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Draw selected points count
        if len(self.selected_points) > 0:
            cv2.putText(frame, f"Selected: {len(self.selected_points)} points", 
                       (20, y_offset + len(instructions) * 25 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def draw_selected_points(self, frame):
        """Draw selected points on the frame."""
        for i, point in enumerate(self.selected_points):
            x, y = int(point[0]), int(point[1])
            # Draw circle
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
            # Draw point number
            cv2.putText(frame, str(i+1), (x+8, y+8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame
    
    def select_points_interactive(self, frame):
        """Interactive point selection from first frame."""
        self.selected_points = []
        self.selecting_points = True
        
        # Set up mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== Point Selection Mode ===")
        print("Click on points you want to track")
        print("Press SPACEBAR to start tracking")
        print("Press 'c' to clear all points")
        print("Press 'q' to quit\n")
        
        while True:
            display_frame = frame.copy()
            
            # Draw instructions
            display_frame = self.draw_instructions(display_frame)
            
            # Draw selected points
            display_frame = self.draw_selected_points(display_frame)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Spacebar
                if len(self.selected_points) > 0:
                    self.selecting_points = False
                    print(f"\nStarting tracking with {len(self.selected_points)} points...")
                    break
                else:
                    print("Please select at least one point before starting!")
            elif key == ord('c'):
                self.selected_points = []
                print("Cleared all selected points")
            elif key == ord('q'):
                self.selecting_points = False
                return False
        
        cv2.destroyWindow(self.window_name)
        return True
    
    def prepare_video_writer(self, video_path, fps, frame_shape, output_path=None):
        """Prepare video writer for saving the output."""
        import os
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        if output_path is None:
            # Generate output filename
            input_filename = Path(video_path).stem
            num_points = len(self.selected_points) if self.selected_points else self.grid_size ** 2
            output_filename = f"{results_dir}/{input_filename}_mesh_{num_points}pts.mp4"
        else:
            output_filename = output_path
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_filename, 
            fourcc, 
            fps, 
            (frame_shape[1], frame_shape[0])
        )
        
        return output_filename
    
    def run_video_file(self, video_path, output_path=None):
        """Run mesh tracking on a video file."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get first frame for point selection
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Interactive point selection
        if not self.select_points_interactive(frame):
            cap.release()
            cv2.destroyAllWindows()
            return
        
        # Initialize tracking with selected points
        self.initialize_tracking(frame, custom_points=self.selected_points if self.selected_points else None)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        
        # Prepare video writer
        output_filename = self.prepare_video_writer(video_path, fps, frame.shape, output_path)
        print(f"Video FPS: {fps}")
        print(f"Saving output to: {output_filename}")
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
            
            # Save frame to video
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.write(frame)
            
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
                
                # Release current video writer
                if hasattr(self, 'video_writer') and self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                
                # Go back to first frame and reinitialize
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    # Re-select points
                    if not self.select_points_interactive(frame):
                        break
                    self.initialize_tracking(frame, custom_points=self.selected_points if self.selected_points else None)
                    # Recreate video writer
                    output_filename = self.prepare_video_writer(video_path, fps, frame.shape, output_path)
        
        # Clean up
        cap.release()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            print(f"Video saved successfully: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description='Mesh-based tracking with LiteTracker')
    parser.add_argument('-w', '--weights', default='weights/scaled_online.pth', 
                       help='Path to model weights (default: weights/scaled_online.pth)')
    parser.add_argument('-s', '--grid_size', type=int, default=10, help='Grid size for tracking points (default: 10)')
    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('-v', '--video', default='assets/stir-5-seq-01.mp4',
                       help='Video file path (default: assets/stir-5-seq-01.mp4, use --no-video for webcam)')
    parser.add_argument('--no-video', action='store_true', 
                       help='Use webcam instead of video file')
    parser.add_argument('-o', '--output', default=None,
                       help='Output video path (default: ./results/<input_name>_mesh_<num_points>pts.mp4)')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'], 
                       help='Device to run on (default: auto, prioritizes Metal/MPS)')
    
    args = parser.parse_args()
    
    # Set OpenMP environment variable first
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Verify weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return
    
    # Create tracker
    tracker = MeshTracker(
        weights_path=args.weights,
        grid_size=args.grid_size,
        device=args.device
    )
    
    # Run tracking
    if args.no_video:
        print(f"Running on webcam (camera {args.camera})")
        tracker.run_webcam(args.camera)
    else:
        # Verify video file exists
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            print("Use --no-video to use webcam instead")
            return
        print(f"Running on video: {args.video}")
        tracker.run_video_file(args.video, output_path=args.output)


if __name__ == "__main__":
    main() 