# 🤟 Sign Language Detection

This project was created during my internship at **Edunet Foundation**.  
It focuses on detecting **sign language letters** (A–Z) using **LSTM (Long Short-Term Memory networks)** and **OpenCV**. Currently, it supports only letter recognition and is in its **development stage**.

---

## 🚀 Project Overview

- 🔹 Recognizes **sign language letters** (not yet words or sentences).  
- 🔹 Built with **LSTM** for sequential modeling and **OpenCV** for real-time video processing.  
- 🔹 Main goal: Make the system more **user-friendly** and extend it to recognize **words, sentences, and even paragraphs** in the future.  

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **TensorFlow / Keras (LSTM)**
- **NumPy & Pandas**
- **Matplotlib** (for data visualization, if used)

---
## ⚡ How It Works

1. The camera captures **hand gestures** using **OpenCV**.  
2. Extracted frames are passed into an **LSTM model**.  
3. The model predicts the **corresponding alphabet letter**.  
4. Future versions will combine predicted letters into **words and sentences**.

---

## 📌 Future Improvements

- 🔹 Extend recognition from **letters → words → sentences → paragraphs**.  
- 🔹 Improve model accuracy using a **larger and more diverse dataset**.  
- 🔹 Build a **user-friendly interface** (desktop app / mobile app / web app).  
- 🔹 Integrate **real-time translation** from sign language to text/speech.  
- 🔹 Add support for **multiple sign languages** (e.g., ASL, ISL, BSL).  
- 🔹 Optimize model for **faster inference** on low-resource devices.   

---

## 🙌 Acknowledgements

- **Edunet Foundation** – for providing me the internship opportunity.  
- **Mentors and trainers** – for their constant support and guidance during development.  
- **Open-source community** – for the tools, libraries, and datasets that made this project possible.  

---

## ▶️ Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/akshatsharma09/Sign_Language_Detection.git
   cd Sign_Language_Detection

