#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizer Module

This module provides 3D visualization capabilities for cricket motion analysis,
including pose rendering, biomechanical insights, and corrective feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import cv2
import logging
import os
from tqdm import tqdm

# Optional imports for advanced 3D rendering
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("open3d not available, falling back to matplotlib for 3D visualization")

try:
    import pyrender
    import trimesh
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    logging.warning("pyrender/trimesh not available, some advanced 3D features will be disabled")

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Class for creating 3D visualizations of cricket motion analysis.
    """
    
    def __init__(self, output_dir, mode='matplotlib', width=1280, height=720, fps=30):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualization outputs
            mode (str): Visualization mode ('matplotlib', 'open3d', 'pyrender')
            width (int): Output video width
            height (int): Output video height
            fps (int): Output video frames per second
        """
        self.output_dir = output_dir
        self.mode = mode
        self.width = width
        self.height = height
        self.fps = fps
        
        # Check if selected mode is available
        if mode == 'open3d' and not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, falling back to matplotlib")
            self.mode = 'matplotlib'
        elif mode == 'pyrender' and not PYRENDER_AVAILABLE:
            logger.warning("pyrender not available, falling back to matplotlib")
            self.mode = 'matplotlib'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define connections between landmarks for skeleton visualization
        self.pose_connections = [
            # Face connections
            (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
            # Upper body connections
            (9, 10), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            # Lower body connections
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
        ]
        
        # Define colors for different body parts
        self.colors = {
            'face': 'cyan',
            'left_arm': 'blue',
            'right_arm': 'green',
            'torso': 'red',
            'left_leg': 'magenta',
            'right_leg': 'yellow'
        }
        
        # Map connections to body parts for coloring
        self.connection_colors = {
            # Face
            (0, 1): self.colors['face'], (0, 4): self.colors['face'],
            (1, 2): self.colors['face'], (2, 3): self.colors['face'],
            (3, 7): self.colors['face'], (4, 5): self.colors['face'],
            (5, 6): self.colors['face'], (6, 8): self.colors['face'],
            # Upper body
            (9, 10): self.colors['torso'],
            (11, 12): self.colors['torso'],
            (11, 13): self.colors['left_arm'], (13, 15): self.colors['left_arm'],
            (12, 14): self.colors['right_arm'], (14, 16): self.colors['right_arm'],
            # Lower body
            (11, 23): self.colors['torso'], (12, 24): self.colors['torso'],
            (23, 24): self.colors['torso'],
            (23, 25): self.colors['left_leg'], (25, 27): self.colors['left_leg'],
            (24, 26): self.colors['right_leg'], (26, 28): self.colors['right_leg']
        }
    
    def visualize_pose_sequence(self, pose_data, analysis_results, video_path, activity_type):
        """
        Create a 3D visualization of the pose sequence with analysis insights.
        
        Args:
            pose_data (list): List of pose data for each frame
            analysis_results (dict): Analysis results from the analyzer
            video_path (str): Path to the original video file
            activity_type (str): Type of cricket activity ('batting', 'bowling', 'fielding')
            
        Returns:
            str: Path to the output visualization video
        """
        logger.info(f"Creating 3D visualization for {activity_type} analysis")
        
        # Select the appropriate visualization method based on mode
        if self.mode == 'matplotlib':
            output_path = self._visualize_with_matplotlib(pose_data, analysis_results, video_path, activity_type)
        elif self.mode == 'open3d':
            output_path = self._visualize_with_open3d(pose_data, analysis_results, video_path, activity_type)
        elif self.mode == 'pyrender':
            output_path = self._visualize_with_pyrender(pose_data, analysis_results, video_path, activity_type)
        else:
            raise ValueError(f"Unsupported visualization mode: {self.mode}")
        
        logger.info(f"Visualization saved to {output_path}")
        return output_path
    
    def _visualize_with_matplotlib(self, pose_data, analysis_results, video_path, activity_type):
        """
        Create visualization using matplotlib.
        
        Args:
            pose_data (list): List of pose data for each frame
            analysis_results (dict): Analysis results from the analyzer
            video_path (str): Path to the original video file
            activity_type (str): Type of cricket activity
            
        Returns:
            str: Path to the output visualization video
        """
        logger.info("Creating visualization with matplotlib")
        
        # Open the original video to get frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video writer
        output_path = os.path.join(self.output_dir, f"{activity_type}_analysis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Process each frame
        for frame_idx in tqdm(range(frame_count), desc="Rendering frames"):
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame to desired dimensions
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Create a figure with two subplots: original video and 3D visualization
            fig = plt.figure(figsize=(self.width/100, self.height/100), dpi=100)
            
            # Original video subplot
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Video")
            ax1.axis('off')
            
            # 3D visualization subplot
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.set_title("3D Pose Analysis")
            
            # Plot 3D pose if available for this frame
            if frame_idx < len(pose_data) and pose_data[frame_idx] is not None:
                self._plot_3d_pose(ax2, pose_data[frame_idx], frame_idx, analysis_results, activity_type)
            
            # Add analysis insights and feedback
            self._add_analysis_text(fig, frame_idx, analysis_results, activity_type)
            
            # Convert matplotlib figure to image
            fig.tight_layout()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Write to video
            out.write(img)
            
            # Close the figure to free memory
            plt.close(fig)
        
        # Release resources
        cap.release()
        out.release()
        
        return output_path
    
    def _visualize_with_open3d(self, pose_data, analysis_results, video_path, activity_type):
        """
        Create visualization using Open3D (more advanced 3D rendering).
        
        Args:
            pose_data (list): List of pose data for each frame
            analysis_results (dict): Analysis results from the analyzer
            video_path (str): Path to the original video file
            activity_type (str): Type of cricket activity
            
        Returns:
            str: Path to the output visualization video
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available, falling back to matplotlib")
            return self._visualize_with_matplotlib(pose_data, analysis_results, video_path, activity_type)
        
        logger.info("Creating visualization with Open3D")
        
        # This is a placeholder for Open3D visualization
        # In a real implementation, we would create a more sophisticated 3D visualization
        # using Open3D's capabilities for mesh rendering and animation
        
        # For now, fall back to matplotlib
        return self._visualize_with_matplotlib(pose_data, analysis_results, video_path, activity_type)
    
    def _visualize_with_pyrender(self, pose_data, analysis_results, video_path, activity_type):
        """
        Create visualization using pyrender (photorealistic 3D rendering).
        
        Args:
            pose_data (list): List of pose data for each frame
            analysis_results (dict): Analysis results from the analyzer
            video_path (str): Path to the original video file
            activity_type (str): Type of cricket activity
            
        Returns:
            str: Path to the output visualization video
        """
        if not PYRENDER_AVAILABLE:
            logger.warning("pyrender not available, falling back to matplotlib")
            return self._visualize_with_matplotlib(pose_data, analysis_results, video_path, activity_type)
        
        logger.info("Creating visualization with pyrender")
        
        # This is a placeholder for pyrender visualization
        # In a real implementation, we would create a more sophisticated 3D visualization
        # using pyrender's capabilities for realistic human body rendering
        
        # For now, fall back to matplotlib
        return self._visualize_with_matplotlib(pose_data, analysis_results, video_path, activity_type)
    
    def _plot_3d_pose(self, ax, pose_frame_data, frame_idx, analysis_results, activity_type):
        """
        Plot 3D pose on the given matplotlib axis.
        
        Args:
            ax (matplotlib.axes.Axes): The 3D axis to plot on
            pose_frame_data (dict): Pose data for a single frame
            frame_idx (int): Frame index
            analysis_results (dict): Analysis results from the analyzer
            activity_type (str): Type of cricket activity
        """
        if 'landmarks' not in pose_frame_data:
            return
        
        landmarks = pose_frame_data['landmarks']
        
        # Extract 3D coordinates
        xs = [lm['x'] for lm in landmarks]
        ys = [lm['y'] for lm in landmarks]
        zs = [lm.get('z', 0) for lm in landmarks]  # Use 0 if z not available
        
        # Plot landmarks
        ax.scatter(xs, ys, zs, c='white', alpha=0.5)
        
        # Plot connections between landmarks to form skeleton
        for connection in self.pose_connections:
            idx1, idx2 = connection
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                x = [landmarks[idx1]['x'], landmarks[idx2]['x']]
                y = [landmarks[idx1]['y'], landmarks[idx2]['y']]
                z = [landmarks[idx1].get('z', 0), landmarks[idx2].get('z', 0)]
                
                color = self.connection_colors.get(connection, 'gray')
                ax.plot(x, y, z, c=color, linewidth=2)
        
        # Highlight key points based on activity type and analysis results
        self._highlight_key_points(ax, landmarks, frame_idx, analysis_results, activity_type)
        
        # Set axis properties for better visualization
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 0.5])
        
        # Set view angle
        ax.view_init(elev=20, azim=-60)
    
    def _highlight_key_points(self, ax, landmarks, frame_idx, analysis_results, activity_type):
        """
        Highlight key points based on activity type and analysis results.
        
        Args:
            ax (matplotlib.axes.Axes): The 3D axis to plot on
            landmarks (list): List of landmark positions
            frame_idx (int): Frame index
            analysis_results (dict): Analysis results from the analyzer
            activity_type (str): Type of cricket activity
        """
        if activity_type == 'batting':
            # Highlight batting-specific key points (e.g., wrists, shoulders for bat angle)
            if len(landmarks) > 15 and len(landmarks) > 16:  # Left and right wrists
                ax.scatter([landmarks[15]['x']], [landmarks[15]['y']], [landmarks[15].get('z', 0)], 
                           c='red', s=100, marker='o', label='Left Wrist')
                ax.scatter([landmarks[16]['x']], [landmarks[16]['y']], [landmarks[16].get('z', 0)], 
                           c='blue', s=100, marker='o', label='Right Wrist')
        
        elif activity_type == 'bowling':
            # Highlight bowling-specific key points (e.g., bowling arm, front foot)
            if len(landmarks) > 14 and len(landmarks) > 16:  # Right elbow and wrist for bowling arm
                ax.scatter([landmarks[14]['x']], [landmarks[14]['y']], [landmarks[14].get('z', 0)], 
                           c='red', s=100, marker='o', label='Bowling Elbow')
                ax.scatter([landmarks[16]['x']], [landmarks[16]['y']], [landmarks[16].get('z', 0)], 
                           c='blue', s=100, marker='o', label='Bowling Wrist')
        
        elif activity_type == 'fielding':
            # Highlight fielding-specific key points (e.g., throwing arm, diving position)
            if len(landmarks) > 14 and len(landmarks) > 16:  # Right elbow and wrist for throwing arm
                ax.scatter([landmarks[14]['x']], [landmarks[14]['y']], [landmarks[14].get('z', 0)], 
                           c='red', s=100, marker='o', label='Throwing Elbow')
                ax.scatter([landmarks[16]['x']], [landmarks[16]['y']], [landmarks[16].get('z', 0)], 
                           c='blue', s=100, marker='o', label='Throwing Wrist')
    
    def _add_analysis_text(self, fig, frame_idx, analysis_results, activity_type):
        """
        Add analysis insights and feedback text to the visualization.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to add text to
            frame_idx (int): Frame index
            analysis_results (dict): Analysis results from the analyzer
            activity_type (str): Type of cricket activity
        """
        # Create a text area for analysis insights
        ax_text = fig.add_axes([0.1, 0.01, 0.8, 0.1])  # [left, bottom, width, height]
        ax_text.axis('off')
        
        # Default text if no specific insights available
        text = f"Frame: {frame_idx}\nActivity: {activity_type.capitalize()}"
        
        # Add activity-specific insights if available
        if activity_type == 'batting' and analysis_results:
            # Extract batting-specific insights
            if 'stance' in analysis_results and analysis_results['stance']:
                stance = analysis_results['stance']
                text += f"\nStance: {stance.get('quality', 'N/A')}"
                if 'feedback' in stance:
                    text += f"\nFeedback: {stance.get('feedback', '')}"
        
        elif activity_type == 'bowling' and analysis_results:
            # Extract bowling-specific insights
            if 'runup' in analysis_results and analysis_results['runup']:
                runup = analysis_results['runup']
                if 'speed_consistency' in runup and runup['speed_consistency'] is not None:
                    consistency = runup['speed_consistency']
                    text += f"\nRun-up Consistency: {consistency:.2f}"
                    
                    # Add feedback based on consistency score
                    if consistency < 0.5:
                        text += "\nFeedback: Improve run-up consistency for better rhythm"
        
        elif activity_type == 'fielding' and analysis_results:
            # Extract fielding-specific insights
            if 'throwing' in analysis_results and analysis_results['throwing']:
                throwing = analysis_results['throwing']
                if 'release_angle' in throwing and throwing['release_angle'] is not None:
                    angle = throwing['release_angle']
                    text += f"\nRelease Angle: {angle:.2f}°"
                    
                    # Add feedback based on release angle
                    if angle < 30:
                        text += "\nFeedback: Increase release angle for better trajectory"
        
        # Add the text to the figure
        ax_text.text(0.5, 0.5, text, ha='center', va='center', fontsize=10)
    
    def create_comparison_visualization(self, original_pose_data, corrected_pose_data, analysis_results, activity_type):
        """
        Create a side-by-side comparison of original and corrected poses.
        
        Args:
            original_pose_data (list): List of original pose data
            corrected_pose_data (list): List of corrected pose data
            analysis_results (dict): Analysis results from the analyzer
            activity_type (str): Type of cricket activity
            
        Returns:
            str: Path to the output comparison video
        """
        logger.info(f"Creating comparison visualization for {activity_type}")
        
        # Prepare output video writer
        output_path = os.path.join(self.output_dir, f"{activity_type}_comparison.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Process each frame
        frame_count = min(len(original_pose_data), len(corrected_pose_data))
        for frame_idx in tqdm(range(frame_count), desc="Rendering comparison frames"):
            # Create a figure with two subplots: original and corrected poses
            fig = plt.figure(figsize=(self.width/100, self.height/100), dpi=100)
            
            # Original pose subplot
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1.set_title("Original Technique")
            
            # Corrected pose subplot
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.set_title("Corrected Technique")
            
            # Plot original pose
            if original_pose_data[frame_idx] is not None:
                self._plot_3d_pose(ax1, original_pose_data[frame_idx], frame_idx, analysis_results, activity_type)
            
            # Plot corrected pose
            if corrected_pose_data[frame_idx] is not None:
                self._plot_3d_pose(ax2, corrected_pose_data[frame_idx], frame_idx, analysis_results, activity_type)
            
            # Add analysis insights and feedback
            self._add_comparison_text(fig, frame_idx, analysis_results, activity_type)
            
            # Convert matplotlib figure to image
            fig.tight_layout()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Write to video
            out.write(img)
            
            # Close the figure to free memory
            plt.close(fig)
        
        # Release resources
        out.release()
        
        return output_path
    
    def _add_comparison_text(self, fig, frame_idx, analysis_results, activity_type):
        """
        Add comparison text to the visualization.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to add text to
            frame_idx (int): Frame index
            analysis_results (dict): Analysis results from the analyzer
            activity_type (str): Type of cricket activity
        """
        # Create a text area for comparison insights
        ax_text = fig.add_axes([0.1, 0.01, 0.8, 0.1])  # [left, bottom, width, height]
        ax_text.axis('off')
        
        # Default text
        text = f"Frame: {frame_idx}\nActivity: {activity_type.capitalize()}"
        
        # Add activity-specific comparison insights
        if activity_type == 'batting' and analysis_results:
            text += "\nKey Improvements:\n- Better bat angle\n- Improved weight transfer\n- More stable head position"
        
        elif activity_type == 'bowling' and analysis_results:
            text += "\nKey Improvements:\n- More consistent run-up\n- Better front foot landing\n- Improved follow-through"
        
        elif activity_type == 'fielding' and analysis_results:
            text += "\nKey Improvements:\n- Better anticipation\n- Improved throwing technique\n- More efficient recovery"
        
        # Add the text to the figure
        ax_text.text(0.5, 0.5, text, ha='center', va='center', fontsize=10)
    
    def generate_report(self, analysis_results, activity_type, output_path=None):
        """
        Generate a detailed analysis report with visualizations.
        
        Args:
            analysis_results (dict): Analysis results from the analyzer
            activity_type (str): Type of cricket activity
            output_path (str, optional): Path to save the report. Defaults to None.
            
        Returns:
            str: Path to the output report
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{activity_type}_report.html")
        
        logger.info(f"Generating analysis report for {activity_type}")
        
        # Create a simple HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{activity_type.capitalize()} Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .insight {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
                .feedback {{ color: #e74c3c; font-weight: bold; }}
                .metric {{ font-weight: bold; color: #2980b9; }}
                img {{ max-width: 100%; height: auto; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{activity_type.capitalize()} Motion Analysis Report</h1>
            
            <div class="section">
                <h2>Overview</h2>
                <p>This report provides a detailed analysis of {activity_type} technique based on video analysis.</p>
            </div>
        """
        
        # Add activity-specific sections
        if activity_type == 'batting':
            html_content += self._generate_batting_report_content(analysis_results)
        elif activity_type == 'bowling':
            html_content += self._generate_bowling_report_content(analysis_results)
        elif activity_type == 'fielding':
            html_content += self._generate_fielding_report_content(analysis_results)
        
        # Close the HTML document
        html_content += """
            <div class="section">
                <h2>Conclusion</h2>
                <p>This analysis provides insights into your technique. Regular practice focusing on the highlighted areas will help improve your performance.</p>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML content to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def _generate_batting_report_content(self, analysis_results):
        """
        Generate batting-specific report content.
        
        Args:
            analysis_results (dict): Batting analysis results
            
        Returns:
            str: HTML content for batting analysis
        """
        html_content = """
            <div class="section">
                <h2>Batting Technique Analysis</h2>
        """
        
        # Add stance analysis
        if 'stance' in analysis_results and analysis_results['stance']:
            stance = analysis_results['stance']
            html_content += """
                <div class="insight">
                    <h3>Stance</h3>
            """
            
            if 'quality' in stance:
                html_content += f"<p>Stance Quality: <span class=\"metric\">{stance['quality']}</span></p>"
            
            if 'feedback' in stance:
                html_content += f"<p class=\"feedback\">Feedback: {stance['feedback']}</p>"
            
            html_content += "</div>"
        
        # Add trigger movement analysis
        if 'trigger_movement' in analysis_results and analysis_results['trigger_movement']:
            trigger = analysis_results['trigger_movement']
            html_content += """
                <div class="insight">
                    <h3>Trigger Movement</h3>
            """
            
            if 'timing' in trigger:
                html_content += f"<p>Timing: <span class=\"metric\">{trigger['timing']}</span></p>"
            
            if 'feedback' in trigger:
                html_content += f"<p class=\"feedback\">Feedback: {trigger['feedback']}</p>"
            
            html_content += "</div>"
        
        # Add bat angle analysis
        if 'bat_angle' in analysis_results and analysis_results['bat_angle']:
            bat_angle = analysis_results['bat_angle']
            html_content += """
                <div class="insight">
                    <h3>Bat Angle</h3>
            """
            
            if 'angle' in bat_angle:
                html_content += f"<p>Angle: <span class=\"metric\">{bat_angle['angle']}°</span></p>"
            
            if 'feedback' in bat_angle:
                html_content += f"<p class=\"feedback\">Feedback: {bat_angle['feedback']}</p>"
            
            html_content += "</div>"
        
        # Close the section
        html_content += "</div>"
        
        return html_content
    
    def _generate_bowling_report_content(self, analysis_results):
        """
        Generate bowling-specific report content.
        
        Args:
            analysis_results (dict): Bowling analysis results
            
        Returns:
            str: HTML content for bowling analysis
        """
        html_content = """
            <div class="section">
                <h2>Bowling Technique Analysis</h2>
        """
        
        # Add run-up analysis
        if 'runup' in analysis_results and analysis_results['runup']:
            runup = analysis_results['runup']
            html_content += """
                <div class="insight">
                    <h3>Run-up</h3>
            """
            
            if 'speed_consistency' in runup and runup['speed_consistency'] is not None:
                consistency = runup['speed_consistency']
                html_content += f"<p>Consistency: <span class=\"metric\">{consistency:.2f}</span></p>"
                
                # Add feedback based on consistency score
                if consistency < 0.5:
                    html_content += "<p class=\"feedback\">Feedback: Improve run-up consistency for better rhythm</p>"
                else:
                    html_content += "<p class=\"feedback\">Feedback: Good run-up consistency</p>"
            
            html_content += "</div>"
        
        # Add front foot landing analysis
        if 'front_foot_landing' in analysis_results and analysis_results['front_foot_landing']:
            landing = analysis_results['front_foot_landing']
            html_content += """
                <div class="insight">
                    <h3>Front Foot Landing</h3>
            """
            
            if 'foot_position_relative_to_body' in landing and landing['foot_position_relative_to_body'] is not None:
                position = landing['foot_position_relative_to_body']
                html_content += f"<p>Position: <span class=\"metric\">({position[0]:.2f}, {position[1]:.2f})</span></p>"
                
                # Add feedback based on foot position
                if position[0] < 0:
                    html_content += "<p class=\"feedback\">Feedback: Front foot landing too close to body</p>"
                else:
                    html_content += "<p class=\"feedback\">Feedback: Good front foot landing position</p>"
            
            html_content += "</div>"
        
        # Add release dynamics analysis
        if 'release' in analysis_results and analysis_results['release']:
            release = analysis_results['release']
            html_content += """
                <div class="insight">
                    <h3>Release Dynamics</h3>
            """
            
            if 'arm_angle_at_release' in release and release['arm_angle_at_release'] is not None:
                angle = release['arm_angle_at_release']
                html_content += f"<p>Arm Angle: <span class=\"metric\">{angle:.2f}°</span></p>"
                
                # Add feedback based on arm angle
                if angle < 90:
                    html_content += "<p class=\"feedback\">Feedback: Increase arm angle at release for better trajectory</p>"
                else:
                    html_content += "<p class=\"feedback\">Feedback: Good arm angle at release</p>"
            
            html_content += "</div>"
        
        # Close the section
        html_content += "</div>"
        
        return html_content
    
    def _generate_fielding_report_content(self, analysis_results):
        """
        Generate fielding-specific report content.
        
        Args:
            analysis_results (dict): Fielding analysis results
            
        Returns:
            str: HTML content for fielding analysis
        """
        html_content = """
            <div class="section">
                <h2>Fielding Technique Analysis</h2>
        """
        
        # Add reaction analysis
        if 'reaction' in analysis_results and analysis_results['reaction']:
            reaction = analysis_results['reaction']
            html_content += """
                <div class="insight">
                    <h3>Anticipation and Reaction</h3>
            """
            
            if 'initial_speed' in reaction and reaction['initial_speed'] is not None:
                speed = reaction['initial_speed']
                html_content += f"<p>Initial Speed: <span class=\"metric\">{speed:.2f}</span></p>"
                
                # Add feedback based on initial speed
                if speed < 0.05:
                    html_content += "<p class=\"feedback\">Feedback: Improve initial reaction speed</p>"
                else:
                    html_content += "<p class=\"feedback\">Feedback: Good initial reaction speed</p>"
            
            html_content += "</div>"
        
        # Add dive mechanics analysis
        if 'dive' in analysis_results and analysis_results['dive']:
            dive = analysis_results['dive']
            html_content += """
                <div class="insight">
                    <h3>Dive Mechanics</h3>
            """
            
            if 'dive_detected' in dive and dive['dive_detected']:
                html_content += "<p>Dive Detected: <span class=\"metric\">Yes</span></p>"
                
                if 'dive_extension' in dive and dive['dive_extension'] is not None:
                    extension = dive['dive_extension']
                    html_content += f"<p>Extension: <span class=\"metric\">{extension:.2f}</span></p>"
                    
                    # Add feedback based on extension
                    if extension < 0.7:
                        html_content += "<p class=\"feedback\">Feedback: Improve body extension during dive</p>"
                    else:
                        html_content += "<p class=\"feedback\">Feedback: Good body extension during dive</p>"
            else:
                html_content += "<p>Dive Detected: <span class=\"metric\">No</span></p>"
            
            html_content += "</div>"
        
        # Add throwing technique analysis
        if 'throwing' in analysis_results and analysis_results['throwing']:
            throwing = analysis_results['throwing']
            html_content += """
                <div class="insight">
                    <h3>Throwing Technique</h3>
            """
            
            if 'throwing_detected' in throwing and throwing['throwing_detected']:
                html_content += "<p>Throw Detected: <span class=\"metric\">Yes</span></p>"
                
                if 'release_angle' in throwing and throwing['release_angle'] is not None:
                    angle = throwing['release_angle']
                    html_content += f"<p>Release Angle: <span class=\"metric\">{angle:.2f}°</span></p>"
                    
                    # Add feedback based on release angle
                    if angle < 30:
                        html_content += "<p class=\"feedback\">Feedback: Increase release angle for better trajectory</p>"
                    else:
                        html_content += "<p class=\"feedback\">Feedback: Good release angle</p>"
            else:
                html_content += "<p>Throw Detected: <span class=\"metric\">No</span></p>"
            
            html_content += "</div>"
        
        # Close the section
        html_content += "</div>"
        
        return html_content