# Sign Language to Text/Audio Converter

This Python project uses a webcam to recognize Indian Sign Language gestures and convert them to text or speech.

## Features

- Real-time sign recognition using webcam
- Converts ISL gestures to text and speech
- Trained using animated ISL video dataset from Kaggle

## Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Place dataset in `dataset/`
4. Run `scripts/preprocess.py` → `scripts/train_model.py`
5. Run real-time recognition: `scripts/predict.py`
