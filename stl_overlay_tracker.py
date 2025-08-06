import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid threading issues
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from stl import mesh
import math
import threading

from src.lite_tracker import LiteTracker
from src.model_utils import get_points_on_a_grid


class STLOverlayTracker:
    def __init__(self, weights_path, device='auto'):
        """
        3D STL overlay tracker that projects 3D models onto video scenes.
        
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
            state_dict = torch.load(f, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 3D model parameters
        self.stl_mesh = None
        self.stl_vertices = None
        self.stl_faces = None
        self.correspondence_points_3d = []
        self.correspondence_points_2d = []
        self.transformation_matrix = None
        
        # Tracking parameters
        self.queries = None
        self.initialized = False
        
        # Performance tracking
        self.processing_times = []
        
        # 3D viewer
        self.fig_3d = None
        self.ax_3d = None
        self.selected_3d_point = None
        
    def load_stl_file(self):
        """Open file dialog and load STL file."""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select STL file",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )
        
        if file_path:
            print(f"Loading STL file: {file_path}")
            self.stl_mesh = mesh.Mesh.from_file(file_path)
            self.stl_vertices = self.stl_mesh.vectors.reshape(-1, 3)
            self.stl_faces = np.arange(len(self.stl_vertices)).reshape(-1, 3)
            
            # Normalize and center the model
            self.normalize_stl_model()
            
            print(f"Loaded STL with {len(self.stl_vertices)} vertices and {len(self.stl_faces)} faces")
            return True
        else:
            print("No STL file selected")
            return False
    
    def normalize_stl_model(self):
        """Normalize and center the STL model."""
        if self.stl_vertices is None:
            return
            
        # Center the model
        center = np.mean(self.stl_vertices, axis=0)
        self.stl_vertices = self.stl_vertices - center
        
        # Scale to fit in unit cube
        max_dim = np.max(np.abs(self.stl_vertices))
        if max_dim > 0:
            self.stl_vertices = self.stl_vertices / max_dim
    
    def create_3d_viewer(self):
        """Create 3D visualization window for STL model."""
        self.fig_3d = plt.figure(figsize=(10, 8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        
        # Plot the STL mesh
        if self.stl_vertices is not None and self.stl_faces is not None:
            # Create triangles for visualization (limit for performance)
            triangles = []
            for face in self.stl_faces[:1000]:  # Limit to first 1000 faces
                triangle = [self.stl_vertices[face[0]], 
                          self.stl_vertices[face[1]], 
                          self.stl_vertices[face[2]]]
                triangles.append(triangle)
            
            # Create 3D collection
            poly3d = Poly3DCollection(triangles, alpha=0.7, facecolor='lightblue', edgecolor='black')
            self.ax_3d.add_collection3d(poly3d)
        
        # Set axis limits
        if self.stl_vertices is not None:
            self.ax_3d.set_xlim([-1, 1])
            self.ax_3d.set_ylim([-1, 1])
            self.ax_3d.set_zlim([-1, 1])
        
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Model - Click to select points')
        
        # Connect click event
        self.fig_3d.canvas.mpl_connect('button_press_event', self.on_3d_click)
        
        plt.show(block=False)
        plt.pause(0.1)  # Give time for window to appear
    
    def on_3d_click(self, event):
        """Handle clicks on the 3D model."""
        if event.inaxes != self.ax_3d:
            return
            
        # Convert 2D click to 3D point
        x2d, y2d = event.xdata, event.ydata
        
        if x2d is None or y2d is None:
            return
        
        # Find closest vertex in 3D model
        if self.stl_vertices is not None:
            closest_idx = self.find_closest_vertex_2d(x2d, y2d)
            if closest_idx is not None:
                point_3d = self.stl_vertices[closest_idx]
                self.selected_3d_point = point_3d
                self.correspondence_points_3d.append(point_3d)
                print(f"Selected 3D point {len(self.correspondence_points_3d)}: {point_3d}")
                
                # Highlight the selected point
                self.ax_3d.scatter(point_3d[0], point_3d[1], point_3d[2], 
                                  color='red', s=100, zorder=10)
                self.fig_3d.canvas.draw()
    
    def find_closest_vertex_2d(self, x2d, y2d):
        """Find the closest vertex to the 2D click point."""
        if self.stl_vertices is None:
            return None
            
        # Simple approach: find vertex with closest 2D projection
        distances = []
        for i, vertex in enumerate(self.stl_vertices):
            # Use X and Y coordinates for 2D distance
            dist = math.sqrt((vertex[0] - x2d)**2 + (vertex[1] - y2d)**2)
            distances.append((dist, i))
        
        # Return closest vertex
        distances.sort()
        return distances[0][1] if distances else None
    
    def select_correspondence_points(self, video_frame):
        """Select corresponding points between 3D model and video."""
        print("Starting correspondence point selection...")
        print("Select 3 pairs of corresponding points:")
        print("1. Click on 3D model window, then click on video window")
        print("2. Repeat for 3 pairs total")
        print("3. Press 'c' to clear points if needed")
        
        # Create 3D viewer
        self.create_3d_viewer()
        
        # Create video window
        cv2.namedWindow('Video - Select correspondence points')
        cv2.setMouseCallback('Video - Select correspondence points', self.on_video_click)
        
        self.current_video_frame = video_frame.copy()
        self.correspondence_points_2d = []
        self.correspondence_points_3d = []
        
        # Show the video frame
        cv2.imshow('Video - Select correspondence points', self.current_video_frame)
        
        # Wait for 3 pairs of points
        pair_count = 0
        while pair_count < 3:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):  # Clear points
                self.correspondence_points_2d = []
                self.correspondence_points_3d = []
                self.current_video_frame = video_frame.copy()
                cv2.imshow('Video - Select correspondence points', self.current_video_frame)
                # Clear 3D plot
                if self.ax_3d:
                    self.ax_3d.clear()
                    # Replot the mesh
                    if self.stl_vertices is not None and self.stl_faces is not None:
                        triangles = []
                        for face in self.stl_faces[:1000]:
                            triangle = [self.stl_vertices[face[0]], 
                                      self.stl_vertices[face[1]], 
                                      self.stl_vertices[face[2]]]
                            triangles.append(triangle)
                        poly3d = Poly3DCollection(triangles, alpha=0.7, facecolor='lightblue', edgecolor='black')
                        self.ax_3d.add_collection3d(poly3d)
                        self.ax_3d.set_xlim([-1, 1])
                        self.ax_3d.set_ylim([-1, 1])
                        self.ax_3d.set_zlim([-1, 1])
                    self.fig_3d.canvas.draw()
                pair_count = 0
                print("Cleared all points")
            
            # Check if we have a complete pair
            if (len(self.correspondence_points_3d) == pair_count + 1 and 
                len(self.correspondence_points_2d) == pair_count + 1):
                pair_count += 1
                print(f"Completed pair {pair_count}/3")
        
        cv2.destroyWindow('Video - Select correspondence points')
        if self.fig_3d:
            plt.close(self.fig_3d)
        
        if len(self.correspondence_points_3d) >= 3 and len(self.correspondence_points_2d) >= 3:
            print("Correspondence points selected successfully!")
            return True
        else:
            print("Not enough correspondence points selected")
            return False
    
    def on_video_click(self, event, x, y, flags, param):
        """Handle clicks on the video frame."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.correspondence_points_2d.append([x, y])
            print(f"Selected video point {len(self.correspondence_points_2d)}: ({x}, {y})")
            
            # Draw the point on the video
            cv2.circle(self.current_video_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.current_video_frame, str(len(self.correspondence_points_2d)), 
                       (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Video - Select correspondence points', self.current_video_frame)
    
    def compute_transformation_matrix(self):
        """Compute transformation matrix from correspondence points."""
        if (len(self.correspondence_points_3d) < 3 or 
            len(self.correspondence_points_2d) < 3):
            return False
        
        # Convert to numpy arrays
        points_3d = np.array(self.correspondence_points_3d, dtype=np.float32)
        points_2d = np.array(self.correspondence_points_2d, dtype=np.float32)
        
        # Compute transformation using least squares
        # This is a simplified approach - in practice you'd use proper camera calibration
        try:
            # For now, use a simple scaling and translation
            # In practice, you'd compute proper 3D-to-2D projection matrix
            self.transformation_matrix = self.compute_simple_transformation(points_3d, points_2d)
            print("Transformation matrix computed successfully")
            return True
        except Exception as e:
            print(f"Error computing transformation: {e}")
            return False
    
    def compute_simple_transformation(self, points_3d, points_2d):
        """Compute a simple transformation matrix."""
        # This is a simplified transformation
        # In practice, you'd use proper camera calibration and projection
        
        # Compute scaling factors
        scale_x = np.std(points_2d[:, 0]) / np.std(points_3d[:, 0])
        scale_y = np.std(points_2d[:, 1]) / np.std(points_3d[:, 1])
        
        # Compute translation
        center_3d = np.mean(points_3d, axis=0)
        center_2d = np.mean(points_2d, axis=0)
        
        # Create transformation matrix (simplified)
        transform = np.eye(4)
        transform[0, 0] = scale_x
        transform[1, 1] = scale_y
        transform[0, 3] = center_2d[0] - scale_x * center_3d[0]
        transform[1, 3] = center_2d[1] - scale_y * center_3d[1]
        
        return transform
    
    def project_3d_to_2d(self, points_3d):
        """Project 3D points to 2D using the transformation matrix."""
        if self.transformation_matrix is None:
            return None
        
        # Apply transformation
        points_homogeneous = np.column_stack([points_3d, np.ones(len(points_3d))])
        points_transformed = (self.transformation_matrix @ points_homogeneous.T).T
        
        # Convert to 2D coordinates
        points_2d = points_transformed[:, :2] / points_transformed[:, 2:]
        
        return points_2d
    
    def draw_3d_overlay(self, frame, coords, visibility, confidence):
        """Draw 3D model overlay on the frame."""
        # Convert tensors to numpy
        coords_np = coords[0, 0].cpu().numpy()  # Shape: (N, 2)
        visibility_np = visibility[0, 0].cpu().numpy()  # Shape: (N,)
        
        # Create a copy of the frame for drawing
        frame_with_overlay = frame.copy()
        
        # Project 3D model to 2D
        if self.stl_vertices is not None and self.transformation_matrix is not None:
            projected_vertices = self.project_3d_to_2d(self.stl_vertices)
            
            if projected_vertices is not None:
                # Draw projected 3D model (simplified - just wireframe)
                for face in self.stl_faces[:100]:  # Limit to first 100 faces for performance
                    if (face[0] < len(projected_vertices) and 
                        face[1] < len(projected_vertices) and 
                        face[2] < len(projected_vertices)):
                        
                        # Get projected triangle vertices
                        v1 = projected_vertices[face[0]]
                        v2 = projected_vertices[face[1]]
                        v3 = projected_vertices[face[2]]
                        
                        # Draw triangle edges
                        if (0 <= v1[0] < frame.shape[1] and 0 <= v1[1] < frame.shape[0] and
                            0 <= v2[0] < frame.shape[1] and 0 <= v2[1] < frame.shape[0] and
                            0 <= v3[0] < frame.shape[1] and 0 <= v3[1] < frame.shape[0]):
                            
                            # Draw triangle outline
                            pts = np.array([[v1[0], v1[1]], [v2[0], v2[1]], [v3[0], v3[1]]], np.int32)
                            cv2.polylines(frame_with_overlay, [pts], True, (0, 255, 255), 1)
                
                # Draw correspondence points
                for i, point in enumerate(self.correspondence_points_2d):
                    cv2.circle(frame_with_overlay, (int(point[0]), int(point[1])), 8, (255, 0, 0), -1)
                    cv2.putText(frame_with_overlay, f"C{i+1}", (int(point[0])+10, int(point[1])-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw tracked points
        for i in range(len(coords_np)):
            if visibility_np[i]:
                x, y = int(coords_np[i][0]), int(coords_np[i][1])
                cv2.circle(frame_with_overlay, (x, y), 3, (0, 255, 0), -1)
        
        # Draw performance info
        fps = self.get_average_fps()
        cv2.putText(frame_with_overlay, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_overlay, f"3D Model Overlay", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame_with_overlay
    
    def process_frame(self, frame):
        """Process a single frame and return tracking results."""
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
    
    def run_video_file(self, video_path):
        """Run 3D overlay tracking on a video file."""
        # Load STL file first
        if not self.load_stl_file():
            print("Failed to load STL file")
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get first frame for correspondence selection
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        
        # Select correspondence points
        if not self.select_correspondence_points(frame):
            print("Failed to select correspondence points")
            return
        
        # Compute transformation matrix
        if not self.compute_transformation_matrix():
            print("Failed to compute transformation matrix")
            return
        
        # Initialize tracking with a simple grid
        H, W = frame.shape[:2]
        grid_pts = get_points_on_a_grid(10, (H, W))
        self.queries = torch.cat([
            torch.ones_like(grid_pts[:, :, :1]) * 0,
            grid_pts,
        ], dim=2).to(self.device)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = 1.0 / fps if fps > 0 else 0.033
        
        print(f"Video FPS: {fps}")
        print("Press 'q' to quit")
        
        # Prepare video writer
        output_filename = self.prepare_video_writer(video_path, fps, frame.shape)
        print(f"Saving output to: {output_filename}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            
            # Process frame
            coords, visibility, confidence, proc_time = self.process_frame(frame)
            
            # Draw 3D overlay
            frame_with_overlay = self.draw_3d_overlay(frame, coords, visibility, confidence)
            
            # Save frame to video
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.write(frame_with_overlay)
            
            # Display
            cv2.imshow('3D STL Overlay Tracker', frame_with_overlay)
            
            # Handle key presses
            key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
            if key == ord('q'):
                break
        
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
        output_filename = f"{results_dir}/{input_filename}_3d_overlay.mp4"
        
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
    parser = argparse.ArgumentParser(description='3D STL Overlay Tracker')
    parser.add_argument('-w', '--weights', required=True, help='Path to model weights')
    parser.add_argument('-v', '--video', required=True, help='Video file path')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'], 
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Set OpenMP environment variable first
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Create tracker
    tracker = STLOverlayTracker(
        weights_path=args.weights,
        device=args.device
    )
    
    # Run tracking
    print(f"Running on video: {args.video}")
    tracker.run_video_file(args.video)


if __name__ == "__main__":
    main() 