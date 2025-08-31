# YOLO Object Detection with OpenCV

This project produces an API endpoint for object detection using the YOLO (You Only Look Once) algorithm in Python. It allows you to detect objects in images using pre-trained YOLO models.

## Features

- Detects multiple objects in images and videos
- Uses YOLOv8+ models
- Displays bounding boxes and class labels

## Requirements

- Python 3.x
- ultralytics
- FastAPI
- Pre-trained YOLO weights and configuration files



## Setup and Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Start up the FastAPI server
    ```bash
    uvicorn image-obj-detection.yolo:app --reload
    ```

3. Open the server URL on the docs path, e.g. http://127.0.0.1:8000/docs

4. Upload your image (or use samples in image-obj-detection\images folder)



## References

- [Model Prediction with Ultralytics YOLO](https://www.ultralytics.com/)

