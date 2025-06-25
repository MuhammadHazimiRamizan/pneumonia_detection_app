# Chest X-Ray Pneumonia Detection using CNN

A Computer Vision project for automatic detection of pneumonia from chest X-ray images using Convolutional Neural Networks (CNN). This system is designed for internal use by medical professionals to assist in radiological interpretation.

---

## 🩺 Overview

This project leverages deep learning techniques to automatically classify chest X-ray images into **Normal**, **Pneumonia**, or **Unknown** categories. The main goal is to assist radiologists or medical personnel in identifying potential pneumonia cases from radiographic images without the need for laboratory examinations.

---

## 🎯 Goals

- Build an accurate and interpretable CNN-based model for pneumonia detection.
- Create a user-friendly web-based interface for uploading and analyzing X-ray images.
- Provide supporting information related to radiological characteristics of pneumonia.
- Deploy the system using Flask and HuggingFace Spaces for accessibility.

---

## 📊 Dataset

- **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- **Content:** 5,863 chest X-ray images categorized into:
  - `NORMAL`
  - `PNEUMONIA` (bacterial or viral)

---

## 🧠 Model Information

| Item           | Detail                                 |
|----------------|----------------------------------------|
| Model Type     | Convolutional Neural Network (CNN)     |
| Architecture   | Sequential CNN (3 Conv layers + Dense) |
| Input Size     | 150 x 150 pixels (RGB)                 |
| Output Classes | 2 (Normal, Pneumonia)                  |
| Activation     | ReLU (hidden), Sigmoid (output)        |
| Optimizer      | Adam                                   |
| Loss Function  | Binary Crossentropy                    |

---

## 🏋️ Training Details

| Parameter          | Value        |
|--------------------|--------------|
| Epochs             | 25           |
| Batch Size         | 32           |
| Learning Rate      | 0.001        |
| Train/Test Split   | 80/20        |
| Data Augmentation  | Yes (rotation, zoom, etc.) |

---

## 📈 Evaluation Metrics

| Metric             | Value        |
|--------------------|--------------|
| Accuracy           | ~96%         |
| Precision          | ~95%         |
| Recall             | ~97%         |
| F1-Score           | ~96%         |

---

## 🚀 Deployment

- **Backend:** Flask
- **Deployment Platform:** HuggingFace Space  
  [Live App](https://huggingface.co/spaces/jimikeren/pneumonia-detection)
- **Features:**
  - Image upload and preview
  - Automated analysis using CNN
  - Prediction confidence with progress bar
  - Radiological characteristics for interpretation

---

## 📌 Project Status

✅ Model Training Completed  
✅ Web UI Finalized  
✅ HuggingFace Deployment Active  
🔜 Optional improvements: model interpretability (e.g., Grad-CAM)

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Flask
- HuggingFace Spaces

---

## 💻 How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/chest-xray-pneumonia-cnn.git
   cd chest-xray-pneumonia-cnn
   
   ```

> 📌 *Note: This system is intended for medical internal use only. The model serves as a supportive tool and not as a replacement for expert radiological diagnosis.*
