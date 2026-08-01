# Backend Handoff

## Entry Point

backend/realtime_recognition.py

## Python Version

Python 3.11+

## Install

pip install -r requirements.txt

## Run

python backend/realtime_recognition.py

## Models

models/
    svm_model.pkl
    scaler.pkl
    label_encoder.pkl

## Configuration

backend/config.py

## Current Features

- Real-time ASL Recognition
- Left & Right Hand Support
- Prediction Smoothing
- Confidence Filtering
- Text Builder
- Text-to-Speech

## Known Limitations

- Static ASL only
- Some letter can occasionally be confused