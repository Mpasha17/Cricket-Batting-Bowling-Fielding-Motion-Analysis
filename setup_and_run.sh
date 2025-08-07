#!/bin/bash

# Cricket Motion Analysis: Setup and Run Script
# This script sets up the environment and runs the cricket motion analysis pipeline

set -e  # Exit on error

echo "Cricket Motion Analysis: Setup and Run"
echo "======================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "\nCreating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "\nActivating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "\nInstalling dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "\nCreating necessary directories..."
mkdir -p data/raw data/processed models results

# Download or prepare test videos
echo "\nSetting up test data..."
python test_data_downloader.py

# Run the analysis
echo "\nRunning cricket motion analysis..."
python run_analysis.py --all --debug

echo "\nAnalysis complete! Results are available in the 'results' directory."
echo "You can also run specific analyses using the following commands:"
echo "  python run_analysis.py --type batting --debug"
echo "  python run_analysis.py --type bowling --debug"
echo "  python run_analysis.py --type fielding --debug"

# Deactivate virtual environment
deactivate