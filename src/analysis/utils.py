#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis Utilities Module

This module provides common utility functions for cricket motion analysis,
including angle calculations, distance measurements, and velocity calculations.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def angle_between(v1, v2):
    """
    Calculate the angle between two vectors in degrees.
    
    Args:
        v1 (numpy.ndarray): First vector
        v2 (numpy.ndarray): Second vector
        
    Returns:
        float: Angle in degrees
    """
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        logger.warning("Cannot calculate angle between zero vectors")
        return None
    
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    # Ensure the ratio is within [-1, 1] to avoid numerical errors
    ratio = max(min(dot_product / norm_product, 1.0), -1.0)
    
    angle_rad = np.arccos(ratio)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        point1 (tuple or numpy.ndarray): First point coordinates (x, y) or (x, y, z)
        point2 (tuple or numpy.ndarray): Second point coordinates (x, y) or (x, y, z)
        
    Returns:
        float: Euclidean distance
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_velocity(point1, point2, time_delta=1.0):
    """
    Calculate the velocity between two points.
    
    Args:
        point1 (tuple or numpy.ndarray): First point coordinates (x, y) or (x, y, z)
        point2 (tuple or numpy.ndarray): Second point coordinates (x, y) or (x, y, z)
        time_delta (float, optional): Time difference between points. Defaults to 1.0.
        
    Returns:
        float: Velocity magnitude
    """
    distance = calculate_distance(point1, point2)
    return distance / time_delta

def calculate_acceleration(velocity1, velocity2, time_delta=1.0):
    """
    Calculate the acceleration between two velocity measurements.
    
    Args:
        velocity1 (float or numpy.ndarray): First velocity
        velocity2 (float or numpy.ndarray): Second velocity
        time_delta (float, optional): Time difference between velocities. Defaults to 1.0.
        
    Returns:
        float: Acceleration magnitude
    """
    if isinstance(velocity1, (int, float)) and isinstance(velocity2, (int, float)):
        return (velocity2 - velocity1) / time_delta
    else:
        return np.linalg.norm(np.array(velocity2) - np.array(velocity1)) / time_delta

def smooth_data(data, window_size=5):
    """
    Apply a simple moving average to smooth data.
    
    Args:
        data (list or numpy.ndarray): Data to smooth
        window_size (int, optional): Size of the smoothing window. Defaults to 5.
        
    Returns:
        numpy.ndarray: Smoothed data
    """
    if len(data) < window_size:
        logger.warning(f"Data length ({len(data)}) is less than window size ({window_size})")
        return np.array(data)
    
    # Convert to numpy array if it's not already
    data_array = np.array(data)
    
    # Handle multi-dimensional data
    if len(data_array.shape) > 1:
        smoothed = np.zeros_like(data_array)
        for i in range(data_array.shape[1]):
            smoothed[:, i] = np.convolve(data_array[:, i], np.ones(window_size)/window_size, mode='same')
        return smoothed
    else:
        # For 1D data
        return np.convolve(data_array, np.ones(window_size)/window_size, mode='same')

def detect_peaks(data, min_height=None, min_distance=1):
    """
    Detect peaks in data.
    
    Args:
        data (list or numpy.ndarray): Data to analyze
        min_height (float, optional): Minimum peak height. Defaults to None.
        min_distance (int, optional): Minimum samples between peaks. Defaults to 1.
        
    Returns:
        list: Indices of detected peaks
    """
    data_array = np.array(data)
    peaks = []
    
    for i in range(1, len(data_array) - 1):
        if data_array[i] > data_array[i-1] and data_array[i] > data_array[i+1]:
            if min_height is None or data_array[i] >= min_height:
                peaks.append(i)
    
    # Filter peaks by min_distance
    if min_distance > 1 and len(peaks) > 1:
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        peaks = filtered_peaks
    
    return peaks

def calculate_joint_angles(landmarks, joint_triplets):
    """
    Calculate angles for specified joints.
    
    Args:
        landmarks (list): List of landmark positions
        joint_triplets (list): List of triplets (point1, joint, point2) defining joints to measure
        
    Returns:
        dict: Dictionary of joint angles
    """
    angles = {}
    
    for name, (p1_idx, joint_idx, p2_idx) in joint_triplets.items():
        if (len(landmarks) > p1_idx and 
            len(landmarks) > joint_idx and 
            len(landmarks) > p2_idx):
            
            p1 = np.array([landmarks[p1_idx]['x'], landmarks[p1_idx]['y']])
            joint = np.array([landmarks[joint_idx]['x'], landmarks[joint_idx]['y']])
            p2 = np.array([landmarks[p2_idx]['x'], landmarks[p2_idx]['y']])
            
            v1 = p1 - joint
            v2 = p2 - joint
            
            angles[name] = angle_between(v1, v2)
        else:
            angles[name] = None
    
    return angles

def normalize_pose(landmarks, reference_points):
    """
    Normalize pose landmarks relative to reference points.
    
    Args:
        landmarks (list): List of landmark positions
        reference_points (tuple): Indices of two reference points for normalization
        
    Returns:
        list: Normalized landmark positions
    """
    ref1_idx, ref2_idx = reference_points
    
    if len(landmarks) <= max(ref1_idx, ref2_idx):
        logger.warning("Reference points not found in landmarks")
        return landmarks
    
    ref1 = np.array([landmarks[ref1_idx]['x'], landmarks[ref1_idx]['y']])
    ref2 = np.array([landmarks[ref2_idx]['x'], landmarks[ref2_idx]['y']])
    
    # Calculate reference vector and its length
    ref_vector = ref2 - ref1
    ref_length = np.linalg.norm(ref_vector)
    
    if ref_length == 0:
        logger.warning("Reference points are the same, cannot normalize")
        return landmarks
    
    # Calculate rotation angle to align reference vector with y-axis
    angle = np.arctan2(ref_vector[1], ref_vector[0]) - np.pi/2
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # Normalize landmarks
    normalized_landmarks = []
    for lm in landmarks:
        if 'x' in lm and 'y' in lm:
            point = np.array([lm['x'], lm['y']])
            
            # Translate relative to first reference point
            translated = point - ref1
            
            # Rotate to align with y-axis
            rotated = rotation_matrix.dot(translated)
            
            # Scale by reference length
            scaled = rotated / ref_length
            
            normalized = {
                'x': scaled[0],
                'y': scaled[1],
                'z': lm.get('z', 0),
                'visibility': lm.get('visibility', 1.0)
            }
            normalized_landmarks.append(normalized)
        else:
            normalized_landmarks.append(lm)
    
    return normalized_landmarks