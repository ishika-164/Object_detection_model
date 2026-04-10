# Object_detection_model
# Design and Implementation of an Intelligent Assistive Robotic Arm with Vision-Based Object Detection and Autonomous Pick-and-Place for Wheelchair Users

## Overview
This project presents a low-cost AI-powered assistive robotic arm designed to help wheelchair users perform object handling tasks independently. The system combines computer vision, machine learning, embedded systems, and robotics to detect objects and perform intelligent pick-and-place operations.

The robotic arm identifies objects such as apple, banana, and orange using a trained YOLOv8 model and performs pick-and-place tasks using a 5-DOF robotic arm controlled through Raspberry Pi 5 and ESP32.

---

## Project Objective
The main objective of this project is to develop an affordable intelligent robotic assistive system that can:

- Detect and recognize objects in real time
- Convert 2D object coordinates into 3D positions using depth mapping
- Perform vision-guided pick-and-place operations
- Assist wheelchair users in daily object handling tasks
- Reduce dependence on caregivers

---

## Features

### Current Features
- Real-time object detection using YOLOv8
- Detection of apple, banana, and orange
- 5-DOF robotic arm movement
- Joystick-controlled manual operation
- 2D to 3D coordinate conversion using depth mapping algorithm
- ESP32-based robotic control

### Future Enhancements
- Autonomous grasping using inverse kinematics
- Voice-controlled operation
- Multi-object tracking
- Advanced depth sensing integration

---

## System Architecture

### Hardware Components
- Raspberry Pi 5
- ESP32 Microcontroller
- USB/Standard Camera
- 5-DOF Robotic Arm
- Servo Motors
- Motor Driver
- Joystick Module
- Power Supply Battery

### Software Components
- Python
- YOLOv8
- OpenCV
- NumPy
- PySerial
- Arduino IDE
- Raspberry Pi OS

---

## Working Principle

1. Camera captures real-time images
2. YOLOv8 detects objects in image frame
3. Objects are classified as apple, banana, or orange
4. 2D coordinates are extracted
5. Depth mapping converts 2D coordinates into 3D coordinates
6. Raspberry Pi processes movement calculations
7. Commands sent to ESP32
8. ESP32 controls robotic arm motors
9. Robotic arm performs pick-and-place action

---

## Object Detection Model

### Model Used:
YOLOv8 Object Detection Model

### Trained Classes:
- Apple
- Banana
- Orange

### Output:
- Object Label
- Confidence Score
- Bounding Box Coordinates

---

## Robotic Arm Details

The robotic arm has 5 Degrees of Freedom:

1. Base Rotation
2. Shoulder Movement
3. Elbow Movement
4. Wrist Adjustment
5. Gripper Open/Close

This enables flexible movement in 3D space for accurate object manipulation.

---

## Technologies Used

### Machine Learning:
- YOLOv8 Deep Learning Detection

### Computer Vision:
- OpenCV

### Embedded Systems:
- Raspberry Pi 5
- ESP32

### Robotics:
- Servo-based 5-DOF Arm

---

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/assistive-robotic-arm.git
cd assistive-robotic-arm
