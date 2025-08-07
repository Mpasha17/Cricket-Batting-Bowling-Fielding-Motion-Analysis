# Cricket Batting, Bowling & Fielding Motion Analysis

A comprehensive AI-powered system for analyzing cricket player movements, providing 3D visualizations with insights and corrective feedback.

## Overview

This project implements a video-based AI model for cricket motion analysis, focusing on three key aspects of cricket:

1. **Batting Mechanics**: Stance, trigger movement, bat angle, timing, shot selection
2. **Bowling Mechanics**: Run-up consistency, load-up, front foot landing, release dynamics
3. **Follow-through**: Balance, wrist motion, shoulder-hip torque
4. **Fielding**: Anticipation reaction, dive mechanics, throwing angle and technique

The system takes video input, performs pose estimation, and outputs a 3D visualization with insights and corrective feedback.

## Features

- **Video Processing**: Extract frames, enhance quality, and detect cricket activities
- **Pose Estimation**: Detect human poses using MediaPipe or OpenPose
- **Biomechanical Analysis**: Analyze batting, bowling, and fielding mechanics
- **3D Visualization**: Render 3D visualizations with insights and corrective feedback
- **Comprehensive Reports**: Generate detailed analysis reports with recommendations

## Project Structure

```
.
├── data/                  # Data directory for videos and processed data
│   ├── raw/               # Raw video files
│   └── processed/         # Processed data files
├── models/                # Trained models and weights
├── notebooks/             # Jupyter notebooks for exploration and visualization
├── src/                   # Source code
│   ├── preprocessing/     # Video preprocessing modules
│   ├── pose_estimation/   # Pose estimation modules
│   ├── analysis/          # Biomechanical analysis modules
│   │   ├── batting/       # Batting analysis
│   │   ├── bowling/       # Bowling analysis
│   │   └── fielding/      # Fielding analysis
│   └── visualization/     # 3D visualization modules
├── main.py                # Main entry point
├── data_downloader.py     # Module for downloading cricket videos
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Cricket-Batting-Bowling-Fielding-Motion-Analysis.git
cd Cricket-Batting-Bowling-Fielding-Motion-Analysis
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script with a video file:

```bash
python main.py --video path/to/video.mp4 --type batting --output results/
```

Options:
- `--video`: Path to the input video file
- `--type`: Type of cricket activity to analyze (batting, bowling, fielding)
- `--output`: Directory to save results
- `--debug`: Enable debug mode (optional)
- `--model`: Pose estimation model to use (mediapipe, openpose) (optional)

### Example

```bash
python main.py --video data/raw/batting_sample.mp4 --type batting --output results/ --model mediapipe
```

## Analysis Components

### Batting Analysis

- **Stance Analysis**: Evaluates the batsman's stance, including foot position, balance, and posture
- **Trigger Movement**: Analyzes the initial movement before the ball is delivered
- **Bat Angle**: Measures the angle of the bat during different phases of the shot
- **Timing Analysis**: Evaluates the timing of the shot relative to the ball delivery
- **Shot Selection**: Classifies the type of shot played

### Bowling Analysis

- **Run-up Analysis**: Evaluates the consistency and speed of the bowler's run-up
- **Load-up Technique**: Analyzes the bowling arm position during the load-up phase
- **Front Foot Landing**: Evaluates the position and angle of the front foot during delivery
- **Release Dynamics**: Analyzes the arm angle, wrist position, and release point
- **Follow-through**: Evaluates the balance and body position after release

### Fielding Analysis

- **Reaction Analysis**: Evaluates the fielder's initial reaction and movement
- **Dive Mechanics**: Analyzes the technique during diving to catch or stop the ball
- **Throwing Technique**: Evaluates the arm angle, release point, and follow-through during throws
- **Recovery Analysis**: Analyzes how quickly and efficiently the fielder recovers after a dive or throw

## Visualization

The system generates several types of visualizations:

1. **3D Pose Visualization**: Shows the detected pose in 3D space
2. **Biomechanical Analysis**: Highlights key joints and angles relevant to the cricket activity
3. **Comparison Visualization**: Side-by-side comparison of the player's technique with an ideal model
4. **HTML Reports**: Detailed analysis reports with visualizations and recommendations

## Dependencies

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- PyTorch
- Open3D (optional)
- Pyrender (optional)

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Future Sportler for the project requirements and evaluation criteria
- MediaPipe and OpenPose for pose estimation capabilities
- The cricket community for domain knowledge and expertise