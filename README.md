# 🛢️ Oil Spill Detection System

> Real-time environmental monitoring using YOLOv8 and 3D CNN — built by Nanda Kishore

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red?logo=streamlit) ![License](https://img.shields.io/badge/license-MIT-green)

## 🌊 Overview

A near real-time oil spill detection system that empowers government authorities and environmental agencies to detect, monitor, and respond to oil spills using advanced AI. Combines YOLOv8 object detection with 3D Convolutional Neural Networks for accurate spill identification from satellite/drone imagery.

## ✨ Features

- 🔍 **Real-time Detection** — YOLOv8-powered object detection on live or uploaded imagery
- 🧠 **3D CNN Analysis** — Temporal pattern recognition for spill spread prediction
- 🌐 **Web Interface** — Interactive Streamlit dashboard with live visualisation
- 🐳 **Docker Ready** — Full containerised deployment with Docker Compose
- ☁️ **Cloud Deployable** — AWS EC2 + Terraform infrastructure as code included
- 📊 **Real-time Tracking** — Monitor spill boundaries and spread over time

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection Model | YOLOv8 (Ultralytics) |
| Deep Learning | Python, 3D CNN, TensorFlow |
| Web Interface | Streamlit |
| Containerisation | Docker, Docker Compose |
| Cloud Infrastructure | AWS EC2, Terraform |
| Database | PostgreSQL |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/kishore-code-create/oil-spill-detection.git
cd oil-spill-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### Docker Deployment
```bash
docker-compose up --build
```

## 📁 Project Structure

```
oil-spill-detection/
├── ODA(OIL)/              # Core detection module
├── OilSpillPortal/        # Web portal backend
├── RealTimeDetection/     # Real-time processing engine
├── streamlit_app.py       # Main Streamlit interface
├── docker-compose.yml     # Container orchestration
├── terraform/             # AWS infrastructure code
└── requirements.txt       # Python dependencies
```

## 👨‍💻 Author

**Nanda Kishore** — AI/ML Engineer  
📧 nandakishoredevarashetti@gmail.com  
🔗 [GitHub](https://github.com/kishore-code-create) | [LinkedIn](https://linkedin.com/in/nanda-kishore-devarashetti)

## 📄 License

MIT License — feel free to use and build upon this project.

