# 👤 Age and Gender Detection using OpenCV and Python

A real-time age and gender detection system built using Python and OpenCV, powered by pre-trained deep learning models. The project uses your webcam to predict a person's age group and gender.

---

## 🧠 Features

- Real-time webcam-based detection
- Uses OpenCV DNN module with Caffe models
- Lightweight and easy to use
- Fully in Python with no extra folders

---

## 📁 Project Structure

AgeGender-Detection/

├── Images

├── AgeGender.py

├── age_deploy.prototxt

├── age_net.caffemodel

├── gender_deploy.prototxt

├── gender_net.caffemodel

├── age_gender.py

├── opencv_face_detector.pbtxt

└── opencv_face_detector_unit8.pb


---

## 🔧 Installation

1. **Clone this repository**

```bash
git clone https://github.com/your-username/Agender.git
cd Agender
```

2. **Run the Project**

```bash
python age_gender.py
```

🧪 Age Groups & Genders
Age groups:
['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

Gender:
['Male', 'Female']
