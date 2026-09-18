# YOLOv8 + ByteTrack Line Crossing Counter

A real-time object detection, tracking, and counting application built using **YOLOv8, ByteTrack, OpenCV, and Streamlit**.

## 📌 Project Overview

This project detects and tracks **people and vehicles** in a video and counts how many objects cross a predefined line.

The system identifies objects such as:

* Person
* Bicycle
* Car
* Motorcycle
* Bus
* Truck

It assigns a unique tracking ID to each detected object and determines whether the object crosses the line **IN** or **OUT**.

## 🚀 Features

* Real-time object detection using YOLOv8
* Object tracking using ByteTrack
* Person and vehicle classification
* IN/OUT counting based on line crossing
* Video file support
* Webcam support
* Live counting dashboard using Streamlit
* Bounding boxes and tracking IDs
* Displays object counts on the video

## 🛠️ Technologies Used

* **Python**
* **YOLOv8** – Object detection
* **ByteTrack** – Object tracking
* **OpenCV** – Video processing
* **NumPy** – Numerical operations
* **Streamlit** – Web interface

## 📂 Project Structure

```text
YOLOv8-ByteTrack-Line-Crossing-Counter/
│
├── ramya.py
├── sai.mp4
├── yolov8n.pt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/YOLOv8-ByteTrack-Line-Crossing-Counter.git
```

Go to the project folder:

```bash
cd YOLOv8-ByteTrack-Line-Crossing-Counter
```

Install the required libraries:

```bash
pip install streamlit opencv-python numpy ultral
```
