# 🧠 COGNITIVE - Brain Tumor Detection System

AI-powered web application for brain tumor detection using Deep Learning.

![COGNITIVE](https://img.shields.io/badge/Status-Active-success)
![React](https://img.shields.io/badge/React-18-blue)
![Django](https://img.shields.io/badge/Django-5.1-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)

---

## 🎯 Overview

COGNITIVE is a full-stack web application that analyzes MRI scans and classifies brain tumors into 4 categories:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

---
## UI Images
<img width="720" height="720" alt="Screenshot (178)" src="https://github.com/user-attachments/assets/1abd1e37-9f9b-4f4f-bb5d-f934a22936ae" />
<img width="720" height="720" alt="Screenshot (179)" src="https://github.com/user-attachments/assets/b249fa3c-641d-4b35-981c-34fce9820a7b" />
<img width="720" height="720" alt="Screenshot (180)" src="https://github.com/user-attachments/assets/d16b46c4-d207-49a4-8e49-8429cdc2a5a9" />
<img width="720" height="720" alt="Screenshot (181)" src="https://github.com/user-attachments/assets/9a5126c0-d6e2-4eec-b27c-47e4ac39cf33" />
<img width="720" height="720" alt="Screenshot (182)" src="https://github.com/user-attachments/assets/98071d6d-a15c-4a41-bbc3-fab477c4a83d" />
<img width="720" height="720" alt="Screenshot (183)" src="https://github.com/user-attachments/assets/8f047939-bbd1-474a-bfda-ee3bd31347e4" />

**PDF file genrated**

<img width="695" height="866" alt="Screenshot 2026-04-02 233830" src="https://github.com/user-attachments/assets/fb3056c9-7c28-4388-b361-4d59fbcd76d0" />


## ✨ Features

- 🔐 Secure user authentication (JWT)
- 📤 MRI scan upload and analysis
- 🤖 AI-powered tumor detection
- 📊 Confidence score display
- 📄 Automatic PDF report generation
- 📜 Scan history with pagination
- 🎨 Modern dark theme UI
- 🌐 3D brain visualization
- ⚡ Real-time predictions

---

## 🏗️ Tech Stack

### Frontend
- React 18 + TypeScript
- Tailwind CSS
- Three.js
- Vite

### Backend
- Django 5.1
- FastAPI
- MySQL
- JWT Authentication

### Machine Learning
- TensorFlow/Keras
- VGG16 (Transfer Learning)
- 17,000+ training images


---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- MySQL 8.0+

### Backend Setup

```bash
cd backend_temp
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python run_server.py
```

Backend runs on: http://localhost:8000

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: http://localhost:5173

### Database Setup

```sql
CREATE DATABASE cognitive_django_db;
```

Update credentials in `backend_temp/cognitive/settings.py`

---

## 📊 Model Performance

- **Overall Accuracy**: 70-75%
- **Pituitary Detection**: 95-98% ✅
- **Meningioma Detection**: 85-90% ✅
- **No Tumor Detection**: 95-99% ✅
- **Glioma Detection**: 60-70% ⚠️

**Confidence Threshold**: 70% minimum

---

## 📁 Project Structure

```
Cognitive/
├── backend_temp/          # Django + FastAPI backend
│   ├── api/              # API endpoints
│   ├── cognitive/        # Django settings
│   └── uploads/          # Uploaded MRI scans
├── frontend/             # React frontend
│   └── src/
│       ├── components/   # React components
│       ├── pages/        # Page components
│       └── services/     # API services
├── best_vgg16.keras      # ML model file
└── README.md
```

---

## 🎓 For College Presentation

See detailed guides:
- `PROJECT_FINAL_SUMMARY.md` - Complete project overview
- `PRESENTATION_CHECKLIST.md` - Presentation preparation guide

---

## ⚠️ Disclaimer

This is an AI-assisted tool for preliminary analysis. **Always consult a qualified medical professional** for diagnosis and treatment.

---

## 📝 License

This project is for educational purposes (College Project).

---

## 🤝 Contributing

This is a college project. Contributions are welcome for educational purposes.

---

**Made with ❤️ for College Project**

### VGG16 Classifier
