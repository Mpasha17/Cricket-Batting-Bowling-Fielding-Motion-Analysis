#!/usr/bin/env python3

"""
Run Cricket Motion Analysis

This script runs the complete cricket motion analysis pipeline, including:
1. Downloading or preparing video data
2. Preprocessing the video
3. Performing pose estimation
4. Analyzing cricket mechanics (batting, bowling, or fielding)
5. Generating 3D visualizations and reports
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.data_downloader import download_from_drive, download_sample_videos, prepare_test_videos
from src.preprocessing.video_processor import extract_frames, enhance_frame
from src.pose_estimation.pose_estimator import MediaPipePoseEstimator
from src.analysis.batting.batting_analyzer import BattingAnalyzer
from src.analysis.bowling.bowling_analyzer import BowlingAnalyzer
from src.analysis.fielding.fielding_analyzer import FieldingAnalyzer
from src.visualization.visualizer import Visualizer


def setup_data(data_dir):
    """Set up data by downloading or preparing videos.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        List of video paths or None if no videos are available
    """
    print("Setting up data...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Try downloading from Google Drive
    videos = download_from_drive()
    
    if not videos or len(videos) == 0:
        # Try downloading sample videos
        videos = download_sample_videos()
        
        if not videos or len(videos) == 0:
            # Prepare test videos
            videos = prepare_test_videos()
            
            if not videos or len(videos) == 0:
                print("No videos available for analysis.")
                return None
    
    print(f"Found {len(videos)} videos for analysis.")
    return videos


def run_analysis(video_path, analysis_type, output_dir, model='mediapipe', debug=False):
    """Run the complete cricket motion analysis pipeline.
    
    Args:
        video_path: Path to the input video
        analysis_type: Type of analysis (batting, bowling, fielding)
        output_dir: Directory to save results
        model: Pose estimation model to use
        debug: Whether to enable debug mode
        
    Returns:
        Path to the generated report or None if analysis fails
    """
    start_time = time.time()
    print(f"\nAnalyzing {os.path.basename(video_path)} for {analysis_type}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Preprocess the video
    print("\nStep 1: Preprocessing video...")
    frames = extract_frames(video_path)
    if debug:
        print(f"  Extracted {len(frames)} frames")
    
    # Step 2: Perform pose estimation
    print("\nStep 2: Performing pose estimation...")
    if model.lower() == 'mediapipe':
        pose_estimator = MediaPipePoseEstimator()
    else:
        print(f"Unsupported pose estimation model: {model}")
        return None
    
    poses = []
    for i, frame in enumerate(frames):
        if debug and i % 10 == 0:
            print(f"  Processing frame {i+1}/{len(frames)}")
        pose = pose_estimator.estimate_pose(frame)
        poses.append(pose)
    
    # Step 3: Perform biomechanical analysis
    print("\nStep 3: Performing biomechanical analysis...")
    analysis_results = {}
    
    if analysis_type.lower() == 'batting':
        analyzer = BattingAnalyzer()
        analysis_results['stance'] = analyzer.analyze_stance(poses)
        analysis_results['trigger_movement'] = analyzer.analyze_trigger_movement(poses)
        analysis_results['bat_angle'] = analyzer.analyze_bat_angle(poses)
        analysis_results['timing'] = analyzer.analyze_timing(poses)
        analysis_results['shot_type'] = analyzer.classify_shot(poses)
    
    elif analysis_type.lower() == 'bowling':
        analyzer = BowlingAnalyzer()
        analysis_results['runup'] = analyzer.analyze_runup(poses)
        analysis_results['loadup'] = analyzer.analyze_loadup(poses)
        analysis_results['front_foot_landing'] = analyzer.analyze_front_foot_landing(poses)
        analysis_results['release_dynamics'] = analyzer.analyze_release_dynamics(poses)
        analysis_results['followthrough'] = analyzer.analyze_followthrough(poses)
    
    elif analysis_type.lower() == 'fielding':
        analyzer = FieldingAnalyzer()
        analysis_results['reaction'] = analyzer.analyze_reaction(poses)
        analysis_results['dive_mechanics'] = analyzer.analyze_dive_mechanics(poses)
        analysis_results['throwing_technique'] = analyzer.analyze_throwing_technique(poses)
        analysis_results['recovery'] = analyzer.analyze_recovery(poses)
    
    else:
        print(f"Unsupported analysis type: {analysis_type}")
        return None
    
    # Step 4: Generate visualizations and report
    print("\nStep 4: Generating visualizations and report...")
    visualizer = Visualizer()
    
    # Generate report
    report_html = visualizer.generate_report(
        activity_type=analysis_type.lower(),
        frames=frames,
        poses=poses,
        analysis_results=analysis_results
    )
    
    # Save report
    report_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{analysis_type}_analysis.html")
    with open(report_path, 'w') as f:
        f.write(report_html)
    
    # Save visualization frames
    vis_dir = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{analysis_type}_visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create 3D visualization frames
    print("  Creating 3D visualization frames...")
    visualizer.visualize_pose_sequence(poses, output_dir=vis_dir, mode='matplotlib')
    
    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")
    print(f"Report saved to: {report_path}")
    
    return report_path


def main():
    """Main function to run the cricket motion analysis pipeline."""
    parser = argparse.ArgumentParser(description="Cricket Motion Analysis")
    parser.add_argument("--video", help="Path to the input video file")
    parser.add_argument("--type", choices=["batting", "bowling", "fielding"], default="batting",
                        help="Type of cricket activity to analyze")
    parser.add_argument("--output", default="results", help="Directory to save results")
    parser.add_argument("--model", choices=["mediapipe", "openpose"], default="mediapipe",
                        help="Pose estimation model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--all", action="store_true", help="Analyze all available videos")
    
    args = parser.parse_args()
    
    # Set up data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    
    if args.all:
        # Analyze all available videos
        videos = setup_data(data_dir)
        if not videos:
            print("No videos available for analysis.")
            return 1
        
        for video in videos:
            run_analysis(
                video_path=video,
                analysis_type=args.type,
                output_dir=args.output,
                model=args.model,
                debug=args.debug
            )
    
    elif args.video:
        # Analyze a specific video
        if not os.path.exists(args.video):
            print(f"Video file not found: {args.video}")
            return 1
        
        run_analysis(
            video_path=args.video,
            analysis_type=args.type,
            output_dir=args.output,
            model=args.model,
            debug=args.debug
        )
    
    else:
        # No video specified, try to use the first available video
        videos = setup_data(data_dir)
        if not videos:
            print("No videos available for analysis.")
            return 1
        
        run_analysis(
            video_path=videos[0],
            analysis_type=args.type,
            output_dir=args.output,
            model=args.model,
            debug=args.debug
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())