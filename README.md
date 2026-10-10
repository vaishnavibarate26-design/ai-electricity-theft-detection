# ⚡ AI Electricity Theft Detection System

An AI-powered system that detects electricity theft by analyzing smart meter consumption data using Machine Learning.

![Accuracy](https://img.shields.io/badge/Accuracy-91.36%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.1-orange)

---

## 🎯 Project Overview

Electricity theft (non-technical losses) costs utility companies over **$96 billion/year** globally. Traditional manual inspection is slow, expensive, and inefficient.

This project uses a **Random Forest Classifier** to automatically flag suspicious users for inspection by analyzing their daily consumption patterns, achieving **91.36% accuracy** on the SGCC dataset.

## 📊 Dataset

- **Source:** [SGCC (State Grid Corporation of China)](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset)
- **Records:** 42,372 electricity consumers
- **Features:** 1,035 daily consumption values per user
- **Labels:** Binary — 0 = Normal (91.5%), 1 = Theft (8.5%)
- **Note:** The dataset (~155 MB) is not included in this repo due to GitHub's file size limit.

## 🚀 Features

- ✅ Random Forest with 100 decision trees
- ✅ `class_weight='balanced'` to handle extreme class imbalance
- ✅ Interactive **Streamlit dashboard** with dark cyber-security theme
- ✅ Confusion matrix, classification report, and feature importance analysis
- ✅ Live training demo with configurable parameters
- ✅ 91.36% accuracy on 4,500 test samples

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML:** Scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit

## 📁 Project Structure
AI_Theft_Project/
├── data/ # Dataset (not uploaded)
├── src/
│ ├── theft_detection.py # ML training script
│ ├── app.py # Streamlit dashboard
│ └── .streamlit/
│ └── config.toml # Dark theme config
├── results/
│ ├── sgcc_confusion_matrix.png
│ ├── feature_importance.png
│ └── Screenshot *.png # Dashboard screenshots
├── requirements.txt
└── .gitignore


## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/vaishnavibarate26-design/ai-electricity-theft-detection.git
cd ai-electricity-theft-detection

2. Install dependencies
bash
pip install -r requirements.txt

3. Download the dataset
Download sgcc_data.csv from Kaggle and place it in data/.

4. Run the ML pipeline
bash
python src/theft_detection.py

5. Launch the dashboard
bash
cd src
streamlit run app.py
Dashboard opens at: http://localhost:8501

📈 Results
Metric	Value
Accuracy	91.36%
Weighted F1-Score	0.890
Users Analyzed	15,000
Thefts Detected	55

Confusion Matrix
Predicted Normal	Predicted Theft
Actual Normal	4,056	47
Actual Theft	342	55

🔬 How It Works
Load Data — 42,372 smart meter records from SGCC dataset

Clean & Prep — Remove missing values, handle class imbalance

Train AI — Random Forest learns from 100 decision trees

Detect Theft — Flag suspicious users for utility inspection

👤 Author
Vaishnavi Barate
ECE (AI & ML) Mini Project | 2026

📜 License
This project is submitted as part of academic coursework. For educational use only.
