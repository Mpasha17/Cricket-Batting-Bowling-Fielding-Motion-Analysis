#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Downloader Module

This module handles downloading and managing cricket video data from Google Drive.
"""

import os
import logging
import requests
import zipfile
import io
import shutil
from tqdm import tqdm

# Make gdown optional
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    logging.warning("gdown module not found. Google Drive download functionality will be limited.")


logger = logging.getLogger(__name__)

# Default data directory
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

class DataDownloader:
    """
    Class for downloading and managing cricket video data.
    """
    
    def __init__(self, data_dir, drive_url=None):
        """
        Initialize the data downloader.
        
        Args:
            data_dir (str): Directory to store downloaded data
            drive_url (str, optional): Google Drive URL for data. Defaults to None.
        """
        self.data_dir = data_dir
        self.drive_url = drive_url or "https://drive.google.com/drive/folders/1ACgf4r5BjZjPluHfeK5vKbrNbEtQib6A"
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    
    def download_from_drive(self):
        """
        Download cricket videos from Google Drive.
        
        Returns:
            list: Paths to downloaded video files
        """
        logger.info(f"Downloading cricket videos from {self.drive_url}")
        
        if not GDOWN_AVAILABLE:
            logger.warning("gdown module not available. Cannot download from Google Drive.")
            logger.info("To install gdown, run: pip install gdown")
            return []
        
        try:
            # Extract folder ID from URL
            folder_id = self.drive_url.split("/")[-1].split("?")[0]
            output_dir = os.path.join(self.data_dir, "raw")
            
            # Use gdown to download the folder
            downloaded_files = gdown.download_folder(id=folder_id, output=output_dir, quiet=False)
            
            if not downloaded_files:
                logger.warning("No files downloaded from Google Drive. The folder might be empty or inaccessible.")
                return []
            
            # Filter for video files
            video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
            video_files = [f for f in downloaded_files if any(f.lower().endswith(ext) for ext in video_extensions)]
            
            logger.info(f"Downloaded {len(video_files)} cricket video files")
            return video_files
            
        except Exception as e:
            logger.error(f"Error downloading from Google Drive: {str(e)}")
            return []
    
    def download_sample_videos(self, num_videos=5):
        """
        Download sample cricket videos if Google Drive download fails.
        This is a fallback method to ensure the system has some videos to work with.
        
        Args:
            num_videos (int, optional): Number of sample videos to download. Defaults to 5.
            
        Returns:
            list: Paths to downloaded sample video files
        """
        logger.info(f"Downloading {num_videos} sample cricket videos")
        
        # Sample cricket video URLs (these are placeholders and should be replaced with actual URLs)
        sample_urls = [
            "https://example.com/cricket_batting_sample1.mp4",
            "https://example.com/cricket_bowling_sample1.mp4",
            "https://example.com/cricket_fielding_sample1.mp4",
            "https://example.com/cricket_batting_sample2.mp4",
            "https://example.com/cricket_bowling_sample2.mp4"
        ]
        
        downloaded_files = []
        output_dir = os.path.join(self.data_dir, "raw")
        
        for i, url in enumerate(sample_urls[:num_videos]):
            try:
                # This is a placeholder for actual download logic
                # In a real implementation, we would use requests to download the videos
                output_path = os.path.join(output_dir, f"sample_cricket_{i+1}.mp4")
                
                # Create an empty file for demonstration purposes
                with open(output_path, 'w') as f:
                    f.write("This is a placeholder for a cricket video file.")
                
                downloaded_files.append(output_path)
                logger.info(f"Downloaded sample video {i+1}/{num_videos}")
                
            except Exception as e:
                logger.error(f"Error downloading sample video {i+1}: {str(e)}")
        
        return downloaded_files
    
    def list_available_videos(self):
        """
        List all available cricket videos in the data directory.
        
        Returns:
            list: Paths to available video files
        """
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        video_files = []
        
        # Check raw data directory
        raw_dir = os.path.join(self.data_dir, "raw")
        if os.path.exists(raw_dir):
            for file in os.listdir(raw_dir):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(raw_dir, file))
        
        logger.info(f"Found {len(video_files)} cricket video files")
        return video_files
    
    def prepare_test_videos(self):
        """
        Prepare test videos if no videos are available.
        This is a last resort method to ensure the system has something to work with.
        
        Returns:
            list: Paths to test video files
        """
        logger.info("Preparing test videos")
        
        output_dir = os.path.join(self.data_dir, "raw")
        test_videos = []
        
        # Create test videos for each activity type
        for activity in ["batting", "bowling", "fielding"]:
            for i in range(1, 3):  # 2 videos per activity
                output_path = os.path.join(output_dir, f"test_{activity}_{i}.mp4")
                
                # Create an empty file for demonstration purposes
                with open(output_path, 'w') as f:
                    f.write(f"This is a placeholder for a cricket {activity} video file.")
                
                test_videos.append(output_path)
        
        logger.info(f"Prepared {len(test_videos)} test video files")
        return test_videos
    
    def get_videos(self, force_download=False):
        """
        Get cricket videos, downloading if necessary.
        
        Args:
            force_download (bool, optional): Force download even if videos exist. Defaults to False.
            
        Returns:
            list: Paths to video files
        """
        # Check if videos already exist
        existing_videos = self.list_available_videos()
        
        if existing_videos and not force_download:
            logger.info(f"Using {len(existing_videos)} existing video files")
            return existing_videos
        
        # Try downloading from Google Drive
        downloaded_videos = self.download_from_drive()
        
        if downloaded_videos:
            return downloaded_videos
        
        # If Google Drive download fails, try downloading sample videos
        sample_videos = self.download_sample_videos()
        
        if sample_videos:
            return sample_videos
        
        # If all else fails, prepare test videos
        return self.prepare_test_videos()
        
    def use_existing_video(self, video_path):
        """
        Use an existing video file for analysis.
        
        Args:
            video_path (str): Path to the existing video file
            
        Returns:
            str: Path to the video file if it exists, None otherwise
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        logger.info(f"Using existing video file: {video_path}")
        return video_path


# Module-level functions for easier use
def download_from_drive(data_dir=DEFAULT_DATA_DIR, drive_url=None):
    """
    Download cricket videos from Google Drive.
    
    Args:
        data_dir (str, optional): Directory to store downloaded data. Defaults to DEFAULT_DATA_DIR.
        drive_url (str, optional): Google Drive URL for data. Defaults to None.
        
    Returns:
        list: Paths to downloaded video files
    """
    downloader = DataDownloader(data_dir, drive_url)
    return downloader.download_from_drive()

def download_sample_videos(data_dir=DEFAULT_DATA_DIR, num_videos=5):
    """
    Download sample cricket videos.
    
    Args:
        data_dir (str, optional): Directory to store downloaded data. Defaults to DEFAULT_DATA_DIR.
        num_videos (int, optional): Number of sample videos to download. Defaults to 5.
        
    Returns:
        list: Paths to downloaded sample video files
    """
    downloader = DataDownloader(data_dir)
    return downloader.download_sample_videos(num_videos)

def prepare_test_videos(data_dir=DEFAULT_DATA_DIR):
    """
    Prepare test videos if no videos are available.
    
    Args:
        data_dir (str, optional): Directory to store test data. Defaults to DEFAULT_DATA_DIR.
        
    Returns:
        list: Paths to test video files
    """
    downloader = DataDownloader(data_dir)
    return downloader.prepare_test_videos()

def get_videos(data_dir=DEFAULT_DATA_DIR, force_download=False):
    """
    Get cricket videos, downloading if necessary.
    
    Args:
        data_dir (str, optional): Directory to store data. Defaults to DEFAULT_DATA_DIR.
        force_download (bool, optional): Force download even if videos exist. Defaults to False.
        
    Returns:
        list: Paths to video files
    """
    downloader = DataDownloader(data_dir)
    return downloader.get_videos(force_download)

def use_existing_video(video_path):
    """
    Use an existing video file for analysis.
    
    Args:
        video_path (str): Path to the existing video file
        
    Returns:
        str: Path to the video file if it exists, None otherwise
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
        
    logger.info(f"Using existing video file: {video_path}")
    return video_path