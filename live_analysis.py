#!/usr/bin/env python3

"""
Live Cricket Video Analysis

This script performs real-time analysis on cricket videos and displays the results
directly on the video as it plays.
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Try to import OpenCV
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("ERROR: OpenCV (cv2) is required for live video analysis.")
    print("Please install it with: pip install opencv-python")
    sys.exit(1)

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("ERROR: MediaPipe is required for pose estimation.")
    print("Please install it with: pip install mediapipe")
    sys.exit(1)

# Import pose estimation and analysis modules
try:
    from src.pose_estimation.pose_estimator import MediaPipePoseEstimator
    from src.analysis.batting.batting_analyzer import analyze as analyze_batting
    from src.analysis.bowling.bowling_analyzer import analyze as analyze_bowling
    from src.analysis.fielding.fielding_analyzer import analyze as analyze_fielding
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError as e:
    ANALYSIS_MODULES_AVAILABLE = False
    print(f"ERROR: Analysis modules not available: {e}")
    print("Please make sure all required modules are installed.")
    sys.exit(1)

def detect_players(frame, analysis_type):
    """
    Detect and identify cricket players in the frame.
    
    Args:
        frame: The video frame to analyze
        analysis_type: Type of analysis (batting, bowling, fielding)
        
    Returns:
        List of detected players with their positions and roles
    """
    # For cricket analysis, define regions of interest based on analysis type
    regions = {
        "batting": {
            "batsman": {"x_min": 0.5, "x_max": 0.8, "y_min": 0.3, "y_max": 0.8},
            "bowler": {"x_min": 0.2, "x_max": 0.5, "y_min": 0.3, "y_max": 0.8}
        },
        "bowling": {
            "bowler": {"x_min": 0.2, "x_max": 0.6, "y_min": 0.3, "y_max": 0.8},
            "batsman": {"x_min": 0.6, "x_max": 0.9, "y_min": 0.3, "y_max": 0.8}
        },
        "fielding": {
            "fielder": {"x_min": 0.1, "x_max": 0.9, "y_min": 0.1, "y_max": 0.9}
        }
    }
    
    # Create a multi-pose detector
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_detector:
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = pose_detector.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        # Determine the role based on position
        role = None
        for potential_role, region in regions[analysis_type].items():
            # Use hip position (average of left and right hip) to determine location
            left_hip_idx = 23  # MediaPipe left hip index
            right_hip_idx = 24  # MediaPipe right hip index
            
            if len(landmarks) > max(left_hip_idx, right_hip_idx):
                left_hip = landmarks[left_hip_idx]
                right_hip = landmarks[right_hip_idx]
                
                hip_center_x = (left_hip['x'] + right_hip['x']) / 2
                hip_center_y = (left_hip['y'] + right_hip['y']) / 2
                hip_visibility = (left_hip['visibility'] + right_hip['visibility']) / 2
                
                # Check if the player is in the expected region for this role
                if (hip_visibility > 0.5 and
                    region["x_min"] <= hip_center_x <= region["x_max"] and
                    region["y_min"] <= hip_center_y <= region["y_max"]):
                    role = potential_role
                    break
        
        # Create a dictionary with the same format as pose_estimator.process_frame()
        return {
            'landmarks': landmarks,
            'pose_landmarks': results.pose_landmarks,  # Keep the original pose_landmarks for drawing
            'role': role,
            'detection_confidence': 1.0
        }

def analyze_frame(frame, frame_count, analysis_type, pose_estimator=None, frame_buffer=None, analysis_results=None):
    """
    Analyze a single frame of cricket video.
    
    Args:
        frame: The video frame to analyze
        frame_count: The current frame number
        analysis_type: Type of analysis (batting, bowling, fielding)
        pose_estimator: MediaPipe pose estimator instance
        frame_buffer: Buffer of recent frames for temporal analysis
        analysis_results: Previous analysis results for tracking
        
    Returns:
        Annotated frame with analysis information and updated analysis results
    """
    # Create a copy of the frame for annotations
    annotated_frame = frame.copy()
    
    # Initialize pose estimator if not provided
    if pose_estimator is None:
        pose_estimator = MediaPipePoseEstimator(static_image_mode=False, model_complexity=1)
    
    # Initialize frame buffer if not provided
    if frame_buffer is None:
        frame_buffer = []
    
    # Initialize analysis results if not provided
    if analysis_results is None:
        analysis_results = {}
    
    # Add frame to buffer (keep last 30 frames)
    frame_buffer.append(frame)
    if len(frame_buffer) > 30:
        frame_buffer.pop(0)
    
    # Detect players in the frame
    player_data = detect_players(frame, analysis_type)
    
    # Use the detected player data if available, otherwise fall back to standard pose estimation
    if player_data and player_data['role']:
        pose_data = player_data
        # Add role information to the frame
        cv2.putText(annotated_frame, f"Detected: {player_data['role'].capitalize()}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # Fall back to standard pose estimation
        pose_data = pose_estimator.process_frame(frame)
        if pose_data:
            cv2.putText(annotated_frame, "Warning: Player role not identified", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add frame counter
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(annotated_frame, f"Time: {timestamp}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add analysis type
    cv2.putText(annotated_frame, f"Analysis: {analysis_type.capitalize()}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw pose landmarks if available
    if pose_data:
        try:
            # Draw pose landmarks directly using MediaPipe drawing utilities
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_pose = mp.solutions.pose
            
            # Create a NormalizedLandmarkList for drawing
            from mediapipe.framework.formats import landmark_pb2
                
            # Handle different pose data formats
            if 'pose_landmarks' in pose_data and pose_data['pose_landmarks']:
                # Use the original MediaPipe pose_landmarks if available
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_data['pose_landmarks'],
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            elif 'landmarks' in pose_data:
                # Convert our custom landmark format to MediaPipe format
                landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark_dict in pose_data['landmarks']:
                    landmark = landmarks_proto.landmark.add()
                    landmark.x = landmark_dict['x']
                    landmark.y = landmark_dict['y']
                    landmark.z = landmark_dict['z']
                    landmark.visibility = landmark_dict['visibility']
                
                # Draw the pose landmarks
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
        except Exception as e:
            # If there's an error drawing the pose, log it but continue
            print(f"Error drawing pose: {e}")
        
        # Perform specific analysis based on type
        if analysis_type == "batting":
            # Analyze batting mechanics
            if frame_count % 10 == 0:  # Analyze every 10 frames to reduce computation
                buffer_pose_data = [pose_estimator.process_frame(f) for f in frame_buffer]
                batting_analysis = analyze_batting(buffer_pose_data)
                if batting_analysis:
                    analysis_results.update(batting_analysis)
            
            # Display batting analysis results
            if 'stance' in analysis_results and analysis_results['stance']:
                stance_info = analysis_results['stance']
                cv2.putText(annotated_frame, f"Stance: {stance_info.get('quality', 'Analyzing...')}", 
                            (frame.shape[1] - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if 'bat_angle' in analysis_results and analysis_results['bat_angle']:
                bat_info = analysis_results['bat_angle']
                angle = bat_info.get('angle', 0)
                cv2.putText(annotated_frame, f"Bat angle: {angle:.1f} degrees", 
                            (frame.shape[1] - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if 'shot_classification' in analysis_results and analysis_results['shot_classification']:
                shot_info = analysis_results['shot_classification']
                shot_type = shot_info.get('shot_type', 'Analyzing...')
                cv2.putText(annotated_frame, f"Shot: {shot_type}", 
                            (frame.shape[1] - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif analysis_type == "bowling":
            # Analyze bowling mechanics
            if frame_count % 10 == 0:  # Analyze every 10 frames to reduce computation
                buffer_pose_data = [pose_estimator.process_frame(f) for f in frame_buffer]
                bowling_analysis = analyze_bowling(buffer_pose_data)
                if bowling_analysis:
                    analysis_results.update(bowling_analysis)
            
            # Display bowling analysis results
            if 'run_up' in analysis_results and analysis_results['run_up']:
                run_up_info = analysis_results['run_up']
                cv2.putText(annotated_frame, f"Run-up: {run_up_info.get('quality', 'Analyzing...')}", 
                            (frame.shape[1] - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if 'release' in analysis_results and analysis_results['release']:
                release_info = analysis_results['release']
                release_point = release_info.get('height', 'Analyzing...')
                cv2.putText(annotated_frame, f"Release: {release_point}", 
                            (frame.shape[1] - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if 'bowling_type' in analysis_results and analysis_results['bowling_type']:
                bowling_info = analysis_results['bowling_type']
                bowling_style = bowling_info.get('style', 'Analyzing...')
                cv2.putText(annotated_frame, f"Style: {bowling_style}", 
                            (frame.shape[1] - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif analysis_type == "fielding":
            # Analyze fielding mechanics
            if frame_count % 10 == 0:  # Analyze every 10 frames to reduce computation
                buffer_pose_data = [pose_estimator.process_frame(f) for f in frame_buffer]
                fielding_analysis = analyze_fielding(buffer_pose_data)
                if fielding_analysis:
                    analysis_results.update(fielding_analysis)
            
            # Display fielding analysis results
            if 'position' in analysis_results and analysis_results['position']:
                position_info = analysis_results['position']
                cv2.putText(annotated_frame, f"Position: {position_info.get('quality', 'Analyzing...')}", 
                            (frame.shape[1] - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if 'movement' in analysis_results and analysis_results['movement']:
                movement_info = analysis_results['movement']
                reaction_time = movement_info.get('reaction_time', 'Analyzing...')
                cv2.putText(annotated_frame, f"Reaction: {reaction_time}", 
                            (frame.shape[1] - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if 'technique' in analysis_results and analysis_results['technique']:
                technique_info = analysis_results['technique']
                technique_quality = technique_info.get('quality', 'Analyzing...')
                cv2.putText(annotated_frame, f"Technique: {technique_quality}", 
                            (frame.shape[1] - 350, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # No pose detected, display message
        cv2.putText(annotated_frame, "No pose detected", (frame.shape[1] - 350, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add general analysis information
    cv2.putText(annotated_frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_frame, pose_estimator, frame_buffer, analysis_results

def run_live_analysis(video_path, analysis_type):
    """
    Run live analysis on a cricket video.
    
    Args:
        video_path: Path to the video file
        analysis_type: Type of analysis (batting, bowling, fielding)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return 1
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    print(f"Running live {analysis_type} analysis...")
    print("Press 'q' to quit")
    
    # Create window for display
    window_name = f"Cricket {analysis_type.capitalize()} Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    # Initialize analysis components
    pose_estimator = MediaPipePoseEstimator(static_image_mode=False, model_complexity=1)
    frame_buffer = []
    analysis_results = {}
    frame_count = 0
    
    # Process video frames
    while True:
        ret, frame = cap.read()
        
        if not ret:
            # End of video or error
            break
        
        # Analyze and annotate the frame
        annotated_frame, pose_estimator, frame_buffer, analysis_results = analyze_frame(
            frame, frame_count, analysis_type, pose_estimator, frame_buffer, analysis_results
        )
        
        # Display the annotated frame
        cv2.imshow(window_name, annotated_frame)
        
        # Increment frame counter
        frame_count += 1
        
        # Control playback speed to match original video
        delay = int(1000 / fps)  # Delay in milliseconds
        
        # Wait for key press (1 ms) and check for quit
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Analysis complete. Processed {frame_count} frames.")
    return 0

def main():
    """
    Main function to run the live cricket video analysis.
    """
    parser = argparse.ArgumentParser(description="Live Cricket Video Analysis")
    parser.add_argument("--video", type=str, default="data/Video-2.mp4",
                        help="Path to the cricket video file")
    parser.add_argument("--type", type=str, choices=["batting", "bowling", "fielding"],
                        default="batting", help="Type of analysis to perform")
    args = parser.parse_args()
    
    # Check if OpenCV is available
    if not OPENCV_AVAILABLE:
        print("Error: OpenCV (cv2) is required for live video analysis.")
        print("Please install it with: pip install opencv-python")
        return 1
    
    # Check if MediaPipe is available
    if not MEDIAPIPE_AVAILABLE:
        print("Error: MediaPipe is required for pose estimation.")
        print("Please install it with: pip install mediapipe")
        return 1
    
    # Check if analysis modules are available
    if not ANALYSIS_MODULES_AVAILABLE:
        print("Error: Analysis modules are required for accurate analysis.")
        print("Please make sure all required modules are installed.")
        return 1
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return 1
    
    print(f"Starting live analysis of: {args.video}")
    print(f"Analysis type: {args.type}")
    print("Using MediaPipe pose estimation for accurate analysis")
    
    # Run the live analysis
    return run_live_analysis(args.video, args.type)

if __name__ == "__main__":
    sys.exit(main())


def draw_annotations(annotated_frame, frame_count, timestamp, analysis_type, pose_data, analysis_results):
    """
    Draw annotations on the frame including timestamp, analysis type, and results.
    
    Args:
        annotated_frame: The frame to draw on
        frame_count: Current frame number
        timestamp: Current timestamp
        analysis_type: Type of analysis (batting, bowling, fielding)
        pose_data: Pose data from MediaPipe
        analysis_results: Results from analysis
        
    Returns:
        Annotated frame with all information drawn
    """
    # Add frame count
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add timestamp
    cv2.putText(annotated_frame, f"Time: {timestamp}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add analysis type
    cv2.putText(annotated_frame, f"Analysis: {analysis_type.capitalize()}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw pose landmarks if available
    if pose_data:
        try:
            # Draw pose landmarks directly using MediaPipe drawing utilities
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_pose = mp.solutions.pose
            
            # Create a NormalizedLandmarkList for drawing
            from mediapipe.framework.formats import landmark_pb2
            
            # Handle different pose data formats
            if 'pose_landmarks' in pose_data and pose_data['pose_landmarks']:
                # Use the original MediaPipe pose_landmarks if available
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_data['pose_landmarks'],
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            elif 'landmarks' in pose_data:
                # Convert our custom landmark format to MediaPipe format
                landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark_dict in pose_data['landmarks']:
                    landmark = landmarks_proto.landmark.add()
                    landmark.x = landmark_dict['x']
                    landmark.y = landmark_dict['y']
                    landmark.z = landmark_dict['z']
                    landmark.visibility = landmark_dict['visibility']
                
                # Draw the pose landmarks
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
        except Exception as e:
            # If there's an error drawing the pose, log it but continue
            print(f"Error drawing pose: {e}")
    
    # Perform specific analysis based on type
    if analysis_type == "batting":
        # Analyze batting mechanics
        if frame_count % 10 == 0:  # Analyze every 10 frames to reduce computation
            # Process frames and ensure we have valid pose data
            buffer_pose_data = []
            for f in frame_buffer:
                processed = pose_estimator.process_frame(f)
                if processed:
                    buffer_pose_data.append(processed)
                    
            # Only analyze if we have enough valid pose data
            if len(buffer_pose_data) > 0:
                print(f"Analyzing frame {frame_count} with {len(buffer_pose_data)} valid poses")
                batting_analysis = analyze_batting(buffer_pose_data)
                if batting_analysis:
                    # Extract only the simple key-value pairs for display
                    for key, value in batting_analysis.items():
                        if key != 'valid_frames':
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    analysis_results[f"{key}_{subkey}"] = subvalue
                            elif isinstance(value, (str, int, float, bool)):
                                analysis_results[key] = value
                    print(f"Frame {frame_count} - New batting analysis: {analysis_results}")
        
        # Display batting analysis results
        y_offset = 150
        for key, value in analysis_results.items():
            cv2.putText(annotated_frame, f"{key}: {value}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
            
    elif analysis_type == "bowling":
        # Analyze bowling mechanics
        if frame_count % 10 == 0:  # Analyze every 10 frames
            # Process frames and ensure we have valid pose data
            buffer_pose_data = []
            for f in frame_buffer:
                processed = pose_estimator.process_frame(f)
                if processed:
                    buffer_pose_data.append(processed)
                    
            # Only analyze if we have enough valid pose data
            if len(buffer_pose_data) > 0:
                print(f"Analyzing frame {frame_count} with {len(buffer_pose_data)} valid poses")
                bowling_analysis = analyze_bowling(buffer_pose_data)
                if bowling_analysis:
                    # Extract only the simple key-value pairs for display
                    for key, value in bowling_analysis.items():
                        if key != 'valid_frames' and isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                analysis_results[f"{key}_{subkey}"] = subvalue
                    print(f"Frame {frame_count} - New bowling analysis: {analysis_results}")
        
        # Display bowling analysis results
        y_offset = 150
        for key, value in analysis_results.items():
            cv2.putText(annotated_frame, f"{key}: {value}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
    
    elif analysis_type == "fielding":
        # Analyze fielding mechanics
        if frame_count % 10 == 0:  # Analyze every 10 frames
            buffer_pose_data = [pose_estimator.process_frame(f) for f in frame_buffer]
            fielding_analysis = analyze_fielding(buffer_pose_data)
            if fielding_analysis:
                analysis_results.update(fielding_analysis)
        
        # Display fielding analysis results
        y_offset = 150
        for key, value in analysis_results.items():
            cv2.putText(annotated_frame, f"{key}: {value}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
    
    return annotated_frame