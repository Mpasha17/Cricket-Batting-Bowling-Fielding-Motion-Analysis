#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fielding Analyzer Module

This module analyzes cricket fielding mechanics using pose data, including:
- Anticipation and reaction
- Dive mechanics
- Throwing angle and technique
- Recovery and follow-through
"""

import numpy as np
import logging
from tqdm import tqdm
from ..utils import angle_between, calculate_distance, calculate_velocity, smooth_data

logger = logging.getLogger(__name__)

# Define key pose landmarks for fielding analysis
KEY_LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

def analyze(pose_data, equipment_data=None, ball_trajectory=None):
    """
    Analyze fielding mechanics from pose data.
    
    Args:
        pose_data (list): List of pose data for each frame
        equipment_data (list, optional): List of equipment detections for each frame
        ball_trajectory (list, optional): Ball trajectory data if available
        
    Returns:
        dict: Dictionary containing fielding analysis results
    """
    logger.info("Analyzing fielding mechanics")
    
    # Filter out frames with no pose data
    valid_frames = [i for i, data in enumerate(pose_data) if data is not None]
    if not valid_frames:
        logger.warning("No valid pose data found for fielding analysis")
        return None
    
    # Extract valid pose data
    valid_pose_data = [pose_data[i] for i in valid_frames]
    
    # Analyze anticipation and reaction
    reaction_analysis = analyze_reaction(valid_pose_data, ball_trajectory)
    
    # Analyze dive mechanics if a dive is detected
    dive_analysis = analyze_dive_mechanics(valid_pose_data)
    
    # Analyze throwing technique
    throwing_analysis = analyze_throwing(valid_pose_data, equipment_data)
    
    # Analyze recovery and follow-through
    recovery_analysis = analyze_recovery(valid_pose_data)
    
    # Compile all analysis results
    analysis_results = {
        'reaction': reaction_analysis,
        'dive': dive_analysis,
        'throwing': throwing_analysis,
        'recovery': recovery_analysis,
        'valid_frames': valid_frames
    }
    
    logger.info("Fielding analysis complete")
    
    return analysis_results

def analyze_reaction(pose_data, ball_trajectory=None):
    """
    Analyze the fielder's anticipation and reaction.
    
    Args:
        pose_data (list): List of valid pose data
        ball_trajectory (list, optional): Ball trajectory data if available
        
    Returns:
        dict: Reaction analysis results
    """
    logger.info("Analyzing fielder's reaction")
    
    # Track the movement of the center of mass over time
    center_of_mass_positions = []
    head_orientations = []
    
    for frame_data in pose_data:
        if frame_data and 'landmarks' in frame_data:
            landmarks = frame_data['landmarks']
            
            # Calculate approximate center of mass (average of hips)
            if (len(landmarks) > KEY_LANDMARKS['left_hip'] and 
                len(landmarks) > KEY_LANDMARKS['right_hip']):
                left_hip = landmarks[KEY_LANDMARKS['left_hip']]
                right_hip = landmarks[KEY_LANDMARKS['right_hip']]
                center_x = (left_hip['x'] + right_hip['x']) / 2
                center_y = (left_hip['y'] + right_hip['y']) / 2
                center_of_mass_positions.append((center_x, center_y))
            else:
                center_of_mass_positions.append(None)
            
            # Calculate head orientation (using nose and midpoint of shoulders)
            if (len(landmarks) > KEY_LANDMARKS['nose'] and
                len(landmarks) > KEY_LANDMARKS['left_shoulder'] and
                len(landmarks) > KEY_LANDMARKS['right_shoulder']):
                
                nose = landmarks[KEY_LANDMARKS['nose']]
                left_shoulder = landmarks[KEY_LANDMARKS['left_shoulder']]
                right_shoulder = landmarks[KEY_LANDMARKS['right_shoulder']]
                
                shoulder_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                shoulder_mid_y = (left_shoulder['y'] + right_shoulder['y']) / 2
                
                # Calculate head orientation vector
                head_vector = (nose['x'] - shoulder_mid_x, nose['y'] - shoulder_mid_y)
                head_orientations.append(head_vector)
            else:
                head_orientations.append(None)
    
    # Calculate reaction time if ball trajectory is available
    reaction_time = None
    if ball_trajectory and len(ball_trajectory) > 0:
        # This is a placeholder for reaction time calculation
        # In a real implementation, we would detect when the fielder first moves in response to the ball
        pass
    
    # Calculate initial movement direction and speed
    initial_direction = None
    initial_speed = None
    
    # Find the first significant movement
    movement_threshold = 0.02  # Threshold for significant movement
    movement_start_frame = None
    
    for i in range(1, len(center_of_mass_positions)):
        if (center_of_mass_positions[i] is not None and 
            center_of_mass_positions[i-1] is not None):
            
            displacement = np.array(center_of_mass_positions[i]) - np.array(center_of_mass_positions[i-1])
            distance = np.linalg.norm(displacement)
            
            if distance > movement_threshold:
                movement_start_frame = i
                
                # Calculate initial direction (angle in degrees from positive x-axis)
                initial_direction = np.degrees(np.arctan2(displacement[1], displacement[0]))
                
                # Calculate initial speed
                initial_speed = distance
                break
    
    # Analyze head tracking (how well the fielder's head follows the ball)
    head_tracking_score = None
    if ball_trajectory and len(head_orientations) > 0:
        # This is a placeholder for head tracking analysis
        # In a real implementation, we would compare head orientation with ball direction
        pass
    
    return {
        'reaction_time': reaction_time,
        'movement_start_frame': movement_start_frame,
        'initial_direction': initial_direction,
        'initial_speed': initial_speed,
        'head_tracking_score': head_tracking_score
    }

def analyze_dive_mechanics(pose_data):
    """
    Analyze the fielder's diving technique if a dive is detected.
    
    Args:
        pose_data (list): List of valid pose data
        
    Returns:
        dict: Dive mechanics analysis results or None if no dive detected
    """
    logger.info("Analyzing fielder's dive mechanics")
    
    # Detect if a dive occurs
    dive_detected = False
    dive_start_frame = None
    dive_end_frame = None
    
    # Track the height of the center of mass over time
    center_heights = []
    
    for frame_data in pose_data:
        if frame_data and 'landmarks' in frame_data:
            landmarks = frame_data['landmarks']
            
            # Calculate approximate center of mass height (average of hips)
            if (len(landmarks) > KEY_LANDMARKS['left_hip'] and 
                len(landmarks) > KEY_LANDMARKS['right_hip']):
                left_hip = landmarks[KEY_LANDMARKS['left_hip']]
                right_hip = landmarks[KEY_LANDMARKS['right_hip']]
                center_y = (left_hip['y'] + right_hip['y']) / 2
                center_heights.append(center_y)
            else:
                center_heights.append(None)
    
    # Smooth the height data to reduce noise
    valid_heights = [h for h in center_heights if h is not None]
    if len(valid_heights) > 5:  # Need enough data points for smoothing
        smoothed_heights = smooth_data(valid_heights, window_size=5)
        
        # Detect significant drop in height (potential dive)
        height_threshold = 0.1  # Threshold for significant height change
        for i in range(1, len(smoothed_heights)):
            height_change = smoothed_heights[i] - smoothed_heights[i-1]
            if height_change > height_threshold:  # Remember that y increases downward in image coordinates
                dive_start_frame = i
                dive_detected = True
                
                # Find dive end (when height stabilizes)
                for j in range(i+1, len(smoothed_heights)):
                    if abs(smoothed_heights[j] - smoothed_heights[j-1]) < height_threshold/2:
                        dive_end_frame = j
                        break
                
                break
    
    if not dive_detected:
        logger.info("No dive detected in the fielding sequence")
        return {
            'dive_detected': False,
            'dive_start_frame': None,
            'dive_end_frame': None,
            'dive_direction': None,
            'dive_extension': None,
            'landing_technique': None
        }
    
    # Analyze dive direction
    dive_direction = None
    if dive_start_frame is not None and dive_start_frame > 0:
        # Calculate direction from movement of center of mass
        if (center_heights[dive_start_frame] is not None and 
            center_heights[dive_start_frame-1] is not None):
            
            # This is a simplified direction calculation
            # In a real implementation, we would use x and y coordinates
            dive_direction = "forward"  # Placeholder
    
    # Analyze body extension during dive
    dive_extension = None
    if dive_start_frame is not None and dive_end_frame is not None:
        # Calculate body extension by measuring distance between extremities
        # This is a placeholder for dive extension analysis
        dive_extension = 0.8  # Placeholder value (0.0 to 1.0, higher is better extension)
    
    # Analyze landing technique
    landing_technique = None
    if dive_end_frame is not None:
        # This is a placeholder for landing technique analysis
        # In a real implementation, we would analyze body position at landing
        landing_technique = "good"  # Placeholder
    
    return {
        'dive_detected': dive_detected,
        'dive_start_frame': dive_start_frame,
        'dive_end_frame': dive_end_frame,
        'dive_direction': dive_direction,
        'dive_extension': dive_extension,
        'landing_technique': landing_technique
    }

def analyze_throwing(pose_data, equipment_data=None):
    """
    Analyze the fielder's throwing technique.
    
    Args:
        pose_data (list): List of valid pose data
        equipment_data (list, optional): List of equipment detections for each frame
        
    Returns:
        dict: Throwing technique analysis results
    """
    logger.info("Analyzing fielder's throwing technique")
    
    # Detect throwing motion
    throwing_detected = False
    throwing_start_frame = None
    throwing_release_frame = None
    throwing_arm = "right"  # Default assumption, should be detected in real implementation
    
    # Track arm positions over time
    arm_positions = []
    
    for frame_data in pose_data:
        if frame_data and 'landmarks' in frame_data:
            landmarks = frame_data['landmarks']
            
            # Track shoulder, elbow, and wrist of throwing arm
            shoulder_idx = KEY_LANDMARKS[f'{throwing_arm}_shoulder']
            elbow_idx = KEY_LANDMARKS[f'{throwing_arm}_elbow']
            wrist_idx = KEY_LANDMARKS[f'{throwing_arm}_wrist']
            
            if (len(landmarks) > shoulder_idx and
                len(landmarks) > elbow_idx and
                len(landmarks) > wrist_idx):
                
                shoulder = landmarks[shoulder_idx]
                elbow = landmarks[elbow_idx]
                wrist = landmarks[wrist_idx]
                
                arm_positions.append({
                    'shoulder': (shoulder['x'], shoulder['y']),
                    'elbow': (elbow['x'], elbow['y']),
                    'wrist': (wrist['x'], wrist['y'])
                })
            else:
                arm_positions.append(None)
    
    # Detect throwing motion by analyzing arm movement patterns
    for i in range(1, len(arm_positions) - 1):
        if (arm_positions[i] is not None and 
            arm_positions[i-1] is not None and
            arm_positions[i+1] is not None):
            
            # Check for backward movement of wrist (cocking phase)
            wrist_prev = np.array(arm_positions[i-1]['wrist'])
            wrist_curr = np.array(arm_positions[i]['wrist'])
            wrist_next = np.array(arm_positions[i+1]['wrist'])
            
            # Calculate wrist movement in x direction
            wrist_movement_x = wrist_curr[0] - wrist_prev[0]
            next_wrist_movement_x = wrist_next[0] - wrist_curr[0]
            
            # Detect transition from backward to forward movement (simplified detection)
            if wrist_movement_x < 0 and next_wrist_movement_x > 0:
                throwing_start_frame = i
                throwing_detected = True
                
                # Find release frame (when wrist reaches maximum forward velocity)
                max_velocity = 0
                for j in range(i+1, min(i+15, len(arm_positions)-1)):  # Look ahead up to 15 frames
                    if arm_positions[j] is not None and arm_positions[j-1] is not None:
                        wrist_j = np.array(arm_positions[j]['wrist'])
                        wrist_prev_j = np.array(arm_positions[j-1]['wrist'])
                        velocity = np.linalg.norm(wrist_j - wrist_prev_j)
                        
                        if velocity > max_velocity:
                            max_velocity = velocity
                            throwing_release_frame = j
                
                break
    
    if not throwing_detected:
        logger.info("No throwing motion detected in the fielding sequence")
        return {
            'throwing_detected': False,
            'throwing_arm': throwing_arm,
            'throwing_start_frame': None,
            'throwing_release_frame': None,
            'arm_angle': None,
            'release_angle': None,
            'throwing_speed': None
        }
    
    # Analyze arm angle during throwing
    arm_angle = None
    if throwing_start_frame is not None and arm_positions[throwing_start_frame] is not None:
        # Calculate angle between upper arm and forearm at start of throw
        shoulder_pos = np.array(arm_positions[throwing_start_frame]['shoulder'])
        elbow_pos = np.array(arm_positions[throwing_start_frame]['elbow'])
        wrist_pos = np.array(arm_positions[throwing_start_frame]['wrist'])
        
        upper_arm_vector = elbow_pos - shoulder_pos
        forearm_vector = wrist_pos - elbow_pos
        
        arm_angle = angle_between(upper_arm_vector, forearm_vector)
    
    # Analyze release angle
    release_angle = None
    if throwing_release_frame is not None and arm_positions[throwing_release_frame] is not None:
        # Calculate angle of arm at release
        shoulder_pos = np.array(arm_positions[throwing_release_frame]['shoulder'])
        wrist_pos = np.array(arm_positions[throwing_release_frame]['wrist'])
        
        throw_vector = wrist_pos - shoulder_pos
        horizontal_vector = np.array([1.0, 0.0])  # Reference horizontal vector
        
        release_angle = angle_between(throw_vector, horizontal_vector)
    
    # Analyze throwing speed
    throwing_speed = None
    if throwing_release_frame is not None and throwing_release_frame > 0:
        if (arm_positions[throwing_release_frame] is not None and 
            arm_positions[throwing_release_frame-1] is not None):
            
            wrist_curr = np.array(arm_positions[throwing_release_frame]['wrist'])
            wrist_prev = np.array(arm_positions[throwing_release_frame-1]['wrist'])
            
            throwing_speed = np.linalg.norm(wrist_curr - wrist_prev)
    
    return {
        'throwing_detected': throwing_detected,
        'throwing_arm': throwing_arm,
        'throwing_start_frame': throwing_start_frame,
        'throwing_release_frame': throwing_release_frame,
        'arm_angle': arm_angle,
        'release_angle': release_angle,
        'throwing_speed': throwing_speed
    }

def analyze_recovery(pose_data):
    """
    Analyze the fielder's recovery after diving or throwing.
    
    Args:
        pose_data (list): List of valid pose data
        
    Returns:
        dict: Recovery analysis results
    """
    logger.info("Analyzing fielder's recovery")
    
    # First check if a dive or throw was detected
    dive_analysis = analyze_dive_mechanics(pose_data)
    throwing_analysis = analyze_throwing(pose_data)
    
    dive_detected = dive_analysis['dive_detected']
    throwing_detected = throwing_analysis['throwing_detected']
    
    if not dive_detected and not throwing_detected:
        logger.info("No dive or throw detected, skipping recovery analysis")
        return {
            'recovery_detected': False,
            'recovery_start_frame': None,
            'recovery_time': None,
            'balance_during_recovery': None
        }
    
    # Determine recovery start frame
    recovery_start_frame = None
    if dive_detected and dive_analysis['dive_end_frame'] is not None:
        recovery_start_frame = dive_analysis['dive_end_frame']
    elif throwing_detected and throwing_analysis['throwing_release_frame'] is not None:
        recovery_start_frame = throwing_analysis['throwing_release_frame']
    
    if recovery_start_frame is None or recovery_start_frame >= len(pose_data) - 5:
        logger.warning("Cannot analyze recovery: start frame not detected or too close to end")
        return {
            'recovery_detected': False,
            'recovery_start_frame': None,
            'recovery_time': None,
            'balance_during_recovery': None
        }
    
    # Track the height and stability of the center of mass during recovery
    center_heights = []
    center_positions = []
    
    for i in range(recovery_start_frame, len(pose_data)):
        frame_data = pose_data[i]
        if frame_data and 'landmarks' in frame_data:
            landmarks = frame_data['landmarks']
            
            # Calculate center of mass position
            if (len(landmarks) > KEY_LANDMARKS['left_hip'] and 
                len(landmarks) > KEY_LANDMARKS['right_hip']):
                left_hip = landmarks[KEY_LANDMARKS['left_hip']]
                right_hip = landmarks[KEY_LANDMARKS['right_hip']]
                center_x = (left_hip['x'] + right_hip['x']) / 2
                center_y = (left_hip['y'] + right_hip['y']) / 2
                center_positions.append((center_x, center_y))
                center_heights.append(center_y)
            else:
                center_positions.append(None)
                center_heights.append(None)
    
    # Detect when recovery is complete (when height stabilizes at standing position)
    recovery_end_frame = None
    recovery_time = None
    
    # Filter out None values
    valid_heights = [(i, h) for i, h in enumerate(center_heights) if h is not None]
    
    if len(valid_heights) > 5:  # Need enough data points for analysis
        indices, heights = zip(*valid_heights)
        smoothed_heights = smooth_data(heights, window_size=5)
        
        # Find when height stabilizes at a low value (standing position)
        height_threshold = 0.02  # Threshold for height stability
        for i in range(1, len(smoothed_heights) - 3):
            # Check if height is stable for a few frames
            if (abs(smoothed_heights[i] - smoothed_heights[i+1]) < height_threshold and
                abs(smoothed_heights[i+1] - smoothed_heights[i+2]) < height_threshold):
                recovery_end_frame = indices[i] + recovery_start_frame
                recovery_time = recovery_end_frame - recovery_start_frame
                break
    
    # Analyze balance during recovery
    balance_during_recovery = None
    if recovery_end_frame is not None:
        # This is a placeholder for balance analysis
        # In a real implementation, we would analyze the stability of movement during recovery
        balance_during_recovery = 0.7  # Placeholder value (0.0 to 1.0, higher is better balance)
    
    return {
        'recovery_detected': True,
        'recovery_start_frame': recovery_start_frame,
        'recovery_end_frame': recovery_end_frame,
        'recovery_time': recovery_time,
        'balance_during_recovery': balance_during_recovery
    }