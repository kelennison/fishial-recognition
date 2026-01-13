# 🐠 Fishial Recognition App

A multi-fish tracking application using YOLOv8 and OpenCV tracking algorithms with interactive pause/resume functionality.

## 🌟 Features

- **Multi-Fish Tracking**: Track multiple fish simultaneously in video streams
- **Interactive UI**: Draw bounding boxes, rename fish, reassign IDs
- **Multiple Trackers**: Choose from 7 OpenCV tracking algorithms
- **Pause/Resume**: Automatic and manual pause states with recovery
- **Data Export**: Export tracking data to CSV with metadata
- **Real-time Visualization**: See tracking progress with visual feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenCV with contrib modules (for tracking algorithms)
- Webcam or video files

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fishial-recognition.git
   cd fishial-recognition


# Run the app
streamlit run app.py

# Alternative: Run and open browser automatically
python -m webbrowser "http://localhost:8501" && streamlit run app.py