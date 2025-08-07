#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cricket Motion Analysis System

This is the main entry point for the Cricket Motion Analysis System.
It provides a command-line interface to analyze cricket videos for
batting, bowling, and fielding mechanics using pose estimation and
biomechanical analysis.

Usage:
    python main.py --video PATH_TO_VIDEO --analysis_type [batting|bowling|fielding] --output OUTPUT_DIR
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Import project modules
from src.preprocessing import video_processor
from src.pose_estimation import pose_estimator
from src.analysis.batting import batting_analyzer
from src.analysis.bowling import bowling_analyzer
from src.analysis.fielding import fielding_analyzer
from src.visualization import visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/cricket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Cricket Motion Analysis System')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--analysis_type', type=str, required=True, 
                        choices=['batting', 'bowling', 'fielding'],
                        help='Type of analysis to perform')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the analysis results')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional visualizations')
    parser.add_argument('--model', type=str, default='mediapipe',
                        choices=['mediapipe', 'openpose'],
                        help='Pose estimation model to use')
    
    return parser.parse_args()

def main():
    """
    Main function to run the cricket motion analysis pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info(f"Starting cricket motion analysis for {args.analysis_type}")
    logger.info(f"Input video: {args.video}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Pose estimation model: {args.model}")
    
    try:
        # Step 1: Preprocess the video
        logger.info("Preprocessing video...")
        frames = video_processor.extract_frames(args.video)
        
        # Step 2: Perform pose estimation
        logger.info("Performing pose estimation...")
        pose_data = pose_estimator.estimate_pose(frames, model=args.model)
        
        # Step 3: Analyze the pose data based on the analysis type
        logger.info(f"Analyzing {args.analysis_type} mechanics...")
        if args.analysis_type == 'batting':
            analysis_results = batting_analyzer.analyze(pose_data)
        elif args.analysis_type == 'bowling':
            analysis_results = bowling_analyzer.analyze(pose_data)
        elif args.analysis_type == 'fielding':
            analysis_results = fielding_analyzer.analyze(pose_data)
        
        # Step 4: Visualize the results
        logger.info("Generating visualizations...")
        output_video_path = os.path.join(args.output, f"{args.analysis_type}_analysis.mp4")
        visualizer.create_visualization(frames, pose_data, analysis_results, output_video_path)
        
        logger.info(f"Analysis complete. Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())