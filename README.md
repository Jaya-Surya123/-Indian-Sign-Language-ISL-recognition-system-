
# ✋ Real-Time Gesture Recognition System

A real-time hand gesture recognition system built using TensorFlow/Keras, MediaPipe, and OpenCV.  
The model is trained on a custom dataset and deployed for live webcam-based prediction.

---

## 📂 Project Structure

```
.
├── gesture_recognition.py   # Real-time webcam prediction
├── training.py              # Model training script
├── model.h5                 # Trained model
├── dataset/                 # Dataset folder (not uploaded if large)
└── README.md
```

---

## 🚀 Features

- Real-time hand tracking using MediaPipe
- Deep learning classification model (TensorFlow/Keras)
- Live webcam gesture prediction
- Custom dataset training support
- Extendable for new gestures

---

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate environment:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install tensorflow opencv-python mediapipe numpy
```

Or use:

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model

Run:

```bash
python training.py
```

This will:
- Load dataset
- Train the CNN model
- Save trained model as `model.h5`

---

## 🎥 Real-Time Gesture Recognition

Run:

```bash
python gesture_recognition.py
```

- Opens webcam
- Detects hand using MediaPipe
- Predicts gesture
- Displays result on screen

Press `q` to quit.

---

## 📊 Dataset Structure

```
dataset/
   ├── A/
   ├── B/
   ├── C/
   └── ...
```

Each folder contains images for one gesture class.
https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl


## 📌 Future Improvements

- Improve model accuracy
- Add data augmentation
- Convert to TensorFlow Lite
- Deploy as web app

---

## 📄 License

This project is open-source and intended for educational purposes.
