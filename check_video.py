#!/usr/bin/env python3

"""
Check Video File

This script checks if the video file exists and can be opened.
"""

import os
import sys

def main():
    """Check if the video file exists and can be opened."""
    # Path to the existing video file
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Video-2.mp4")
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return 1
    
    print(f"Video file exists: {video_path}")
    
    # Try to open the file
    try:
        with open(video_path, 'rb') as f:
            # Read the first few bytes to check if it's a valid file
            header = f.read(16)
            print(f"Successfully opened video file. First few bytes: {header.hex()}")
            
            # Get file size
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            print(f"Video file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
            
            return 0
    except Exception as e:
        print(f"Error opening video file: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())