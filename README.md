# ✍ Air Writing Recognition

Real-time **English alphabet recognition** using hand gestures in the air, powered by **MediaPipe Hand Landmarker** and a **K-Nearest Neighbors (KNN)** machine learning model.

---

## 📖 Overview

Air Writing Recognition allows users to write English letters in the air using only their **index finger**, tracked live through a webcam.

After drawing a letter, a predefined hand gesture triggers the model to recognize and predict the drawn character—eliminating the need for a keyboard, mouse, or touchscreen.

---

## 🎯 Features

- ✋ Real-time hand tracking using **MediaPipe Tasks (Hand Landmarker)**
- ✍ Air drawing with your index finger
- 🖐 Gesture-based controls (no buttons required)
- 🤖 KNN-based English alphabet recognition
- 📊 HOG (Histogram of Oriented Gradients) feature extraction
- 📈 Live prediction confidence score
- 🌐 Streamlit web interface with real-time webcam feed and overlays

---

## 🖐 Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ Index Finger | Draw |
| ☝️☝️☝️ Index + Middle + Ring Fingers | Predict |
| ✋ Open Palm | Clear Canvas |

> **Note**
>
> - **Right Hand:** Show the **inside of your palm**.
> - **Left Hand:** Show the **outside of your palm**.
>
> This helps MediaPipe correctly identify the intended gestures.

---

## 🧠 Model Details

| Component | Details |
|-----------|---------|
| Model | K-Nearest Neighbors (KNN) |
| Feature Extraction | Histogram of Oriented Gradients (HOG) |
| Hand Tracking | MediaPipe Tasks (Hand Landmarker) |
| Input Size | 34 × 34 pixels |
| Classes | 26 (A–Z uppercase English alphabets) |

The final KNN model was selected after comparing it against:

- Logistic Regression
- Decision Tree
- Random Forest
- Gaussian Naive Bayes

Hyperparameter tuning was performed using **GridSearchCV**, where KNN achieved the best overall performance based on:

- Accuracy
- Precision
- Recall
- F1-score

---

## 📁 Project Structure

```text
AIR_WRITING/
├── air/                          # Virtual environment (ignored by Git)
├── backend/
│   ├── canvas.py                 # Drawing canvas logic
│   ├── frame_processor.py        # Streamlit-WebRTC frame processor
│   ├── hand_tracker.py           # Hand tracking & gesture detection
│   ├── predictor.py              # Loads trained model & predicts letters
│   └── preprocessing.py          # Image preprocessing & HOG extraction
│
├── english_alphabets/            # Dataset (A-Z image folders)
│
├── models/
│   ├── hand_landmarker.task
│   ├── hog_params.pkl
│   ├── knn_gesture_model.pkl
│   └── label_encoder.pkl
│
├── .gitignore
├── air_writing.ipynb             # Complete ML pipeline
├── app.py                        # Streamlit application
├── main.py                       # OpenCV desktop application
├── requirements.txt
└── style.css                     # Custom Streamlit styling
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Webcam

---

### Clone the Repository

```bash
git clone https://github.com/premaditya/Air-Writing.git
cd Air-Writing
```

---

### Create a Virtual Environment

**Windows**

```bash
python -m venv air
air\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv air
source air/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

### Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

---

### Standalone OpenCV Desktop Version

```bash
python main.py
```

Keyboard shortcuts:

- **C** → Clear canvas
- **Q** → Quit application

---

## 🏋 Training Your Own Model

The complete training workflow is available in:

```
air_writing.ipynb
```

The notebook includes:

- Dataset loading
- Exploratory Data Analysis (EDA)
- Image preprocessing
- HOG feature extraction
- Model comparison
- Hyperparameter tuning using GridSearchCV
- Model evaluation
- Confusion matrix
- Saving trained models

---

## 🛠 Tech Stack

- Python
- OpenCV
- MediaPipe Tasks
- scikit-learn
- scikit-image
- Streamlit
- streamlit-webrtc
- Joblib

---

## ⚠ Known Limitations

- Predictions may be incorrect when:
  - Gestures are unclear
  - Hand movement is too fast
  - Lighting conditions are poor
- Best performance is achieved by drawing slowly under good lighting.
- Currently supports **one hand at a time**.

---

## 📄 License

This project is open source and available for educational and research purposes.

Feel free to use, modify, and build upon it.