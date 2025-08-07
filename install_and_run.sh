#!/bin/bash

# Cricket Motion Analysis Installation and Run Script

echo "=== Cricket Motion Analysis System ==="
echo "This script will install the required dependencies and run the analysis."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if the video file exists
VIDEO_PATH="data/Video-2.mp4"
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found at $VIDEO_PATH"
    echo "Please make sure the video file exists in the data directory."
    exit 1
fi

# Run the analysis
echo "Running cricket video analysis..."
echo "This may take some time depending on your system's performance."
echo

# Ask user which type of analysis to run
echo "Which type of analysis would you like to run?"
echo "1. Batting Analysis"
echo "2. Bowling Analysis"
echo "3. Fielding Analysis"
echo "4. All Analyses"
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        python analyze_video.py --type batting --debug
        ;;
    2)
        python analyze_video.py --type bowling --debug
        ;;
    3)
        python analyze_video.py --type fielding --debug
        ;;
    4)
        python analyze_video.py --type batting --debug
        python analyze_video.py --type bowling --debug
        python analyze_video.py --type fielding --debug
        ;;
    *)
        echo "Invalid choice. Running batting analysis by default."
        python analyze_video.py --type batting --debug
        ;;
esac

echo
echo "Analysis complete! Check the results directory for the analysis reports."

# Deactivate virtual environment
deactivate