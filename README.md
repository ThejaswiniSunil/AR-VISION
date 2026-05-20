# NEXTGEN VISION AI

AI-powered Real-Time AR Vision Enhancement System for adverse weather driving conditions.

NEXTGEN VISION AI combines:
- Real-time image dehazing
- YOLOv8 hazard detection
- AR-style navigation overlays
- Live weather intelligence
- Trip analytics dashboard

Built using Streamlit, PyTorch, OpenCV, and YOLOv8.

---

## Features

### Real-Time Dehazing
Uses a hybrid:
- DCP (Dark Channel Prior)
- ResNet refinement network

to restore visibility in:
- fog
- haze
- smoke
- low-visibility road conditions

---

### YOLOv8 Hazard Detection
Detects driving-related objects such as:
- Cars
- Trucks
- Buses
- Pedestrians
- Traffic lights
- Stop signs
- Motorcycles

with AR-style overlays.

---

### Navigation HUD
Displays:
- Navigation guidance
- Distance remaining
- Route overlays
- AR-inspired directional interface

---

### Weather Integration
Live weather monitoring using OpenWeather API:
- Fog detection
- Rain alerts
- Visibility estimation
- Driving risk assessment

---

### Trip Analytics
Generates post-trip analytics including:
- Visibility improvement %
- Fog severity
- Total hazards detected
- Object frequency
- Detection graphs

---

# Tech Stack

- Python
- Streamlit
- PyTorch
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- Pandas
- PIL
- gdown

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/ThejaswiniSunil/AR-VISION.git
cd AR-VISION
