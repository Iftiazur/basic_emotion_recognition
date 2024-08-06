# Emotion Detection System

This project is an Emotion Detection System implemented in Python. It uses computer vision and deep learning techniques to detect and classify emotions in real-time from webcam feed.

## Features

- Real-time emotion detection from webcam feed
- User interface for starting emotion detection sessions
- Emotion frequency visualization
- Saving emotion detection results for individual users

## Components

1. **Main Application (main.py)**
   - Implements a simple graphical user interface using tkinter
   - Allows users to enter their name and start an emotion detection session

2. **Emotion Detection (detection.py)**
   - Loads a pre-trained convolutional neural network model
   - Performs real-time face detection and emotion classification

3. **Emotion Detection Session (start_detection.py)**
   - Manages the emotion detection process
   - Handles webcam feed and user interactions during the session
   - Saves emotion frequency graphs for individual users

4. **Graph Visualization (show_graph.py)**
   - Provides functionality to display emotion frequency graphs

5. **Model Training (training.py)**
   - Defines and trains the convolutional neural network for emotion detection
   - Uses image data generators for efficient training
   - Saves the trained model weights

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- Tkinter
- Matplotlib
- NumPy
