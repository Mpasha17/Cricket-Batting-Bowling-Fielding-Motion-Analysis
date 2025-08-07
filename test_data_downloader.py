#!/usr/bin/env python3

"""
Test script for the data_downloader module.

This script tests the functionality of the data_downloader module by attempting to
download cricket videos from the provided Google Drive link or fallback to sample videos.
"""

import os
import sys
from src.data_downloader import download_from_drive, download_sample_videos, prepare_test_videos, use_existing_video


def main():
    """Test the data_downloader module functionality."""
    print("Testing Cricket Motion Analysis Data Downloader")
    print("-" * 50)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print()
    
    # Check for existing video file
    existing_video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Video-2.mp4")
    if os.path.exists(existing_video_path):
        print(f"Found existing video file: {existing_video_path}")
        print("Using existing video for analysis.")
        return 0
    
    # Try downloading from Google Drive
    print("Attempting to download videos from Google Drive...")
    drive_videos = download_from_drive()
    
    if drive_videos and len(drive_videos) > 0:
        print(f"Successfully downloaded {len(drive_videos)} videos from Google Drive:")
        for video in drive_videos:
            print(f"  - {os.path.basename(video)}")
    else:
        print("Failed to download videos from Google Drive.")
        
        # Try downloading sample videos
        print("\nAttempting to download sample videos...")
        sample_videos = download_sample_videos()
        
        if sample_videos and len(sample_videos) > 0:
            print(f"Successfully downloaded {len(sample_videos)} sample videos:")
            for video in sample_videos:
                print(f"  - {os.path.basename(video)}")
        else:
            print("Failed to download sample videos.")
            
            # Prepare test videos
            print("\nPreparing test videos...")
            test_videos = prepare_test_videos()
            
            if test_videos and len(test_videos) > 0:
                print(f"Successfully prepared {len(test_videos)} test videos:")
                for video in test_videos:
                    print(f"  - {os.path.basename(video)}")
            else:
                print("Failed to prepare test videos.")
                print("No videos available for analysis.")
                return 1
    
    print("\nVideos are ready for analysis!")
    return 0


if __name__ == "__main__":
    sys.exit(main())