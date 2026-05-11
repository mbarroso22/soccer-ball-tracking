# Soccer Ball Tracking and Temporal Stabilization Using YOLOv8 and Kalman Filtering

## Overview

This project investigates soccer ball tracking in broadcast soccer footage using a lightweight tracking-by-detection pipeline. The system combines pretrained YOLOv8 object detection with temporal post-processing techniques including motion smoothing and confidence-gated Kalman filtering.

The project evaluates tracking performance using the SoccerNet tracking dataset and tests deployment on full English Premier League broadcast footage provided through the SoccerNet research agreement.

The main goal of the project is to analyze whether temporal stabilization can improve soccer ball trajectory continuity after missed detections from a pretrained detector.

---

## Pipeline

SoccerNet Tracking Clips / Broadcast Footage  
→ YOLOv8 Object Detection  
→ Sports Ball Extraction  
→ Motion Smoothing  
→ Kalman Filtering  
→ Confidence-Gated Temporal Stabilization  
→ Ground-Truth Evaluation  
→ Broadcast Deployment Testing

---

## Repository Structure

```text
soccer-ball-tracking/
│
├── src/                    # Full tracking pipeline scripts
├── final_assets/
│   ├── demo_videos/        # Final demo videos
│   ├── figures/            # Figures used in report
│   └── tables/             # Final evaluation tables
│
├── results/                # CSV evaluation outputs
├── paper/                  # Final paper PDF and LaTeX source
├── requirements.txt
├── README.md
└── .gitignore