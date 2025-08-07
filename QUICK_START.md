# Cricket Motion Analysis - Quick Start Guide

## Overview

This system analyzes cricket batting, bowling, and fielding techniques using video processing and pose estimation. It provides biomechanical analysis and visualization of cricket motions.

## Quick Start

### Option 1: Run Simple Analysis (No Dependencies Required)

To quickly see a simulated analysis of your cricket video without installing dependencies:

```bash
python simple_analysis.py
```

This will generate a basic analysis report in the `results` directory.

### Option 2: Install Dependencies and Run Full Analysis

For the complete analysis with all features:

1. Run the installation script:

```bash
./install_and_run.sh
```

This script will:
- Create a virtual environment
- Install all required dependencies
- Run the analysis on your video file

### Option 3: Manual Installation and Running

If you prefer to install dependencies manually:

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the analysis:

```bash
python analyze_video.py --type batting --debug
```

You can replace `batting` with `bowling` or `fielding` to analyze different aspects.

## Video Requirements

The system expects your cricket video file to be located at `data/Video-2.mp4`. Make sure your video:

- Shows clear footage of the cricket player
- Has good lighting conditions
- Is in MP4 format
- Is of reasonable length (30 seconds to 2 minutes recommended)

## Results

After running the analysis, check the `results` directory for:

- Analysis reports
- Visualizations
- Biomechanical data

## Troubleshooting

If you encounter issues:

1. Make sure your video file exists at the correct location
2. Check that all dependencies are properly installed
3. Try running the simple analysis first to verify basic functionality

## Next Steps

For more detailed information about the system, refer to the main `README.md` file.