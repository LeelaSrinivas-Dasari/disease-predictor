# 🩺 Disease Prediction Web App

A Machine Learning-based web application that predicts diseases based on user-selected symptoms. Built using **Python, Streamlit, and Ensemble Learning**, this app provides predictions along with confidence scores, severity, precautions, and explainability using SHAP.

---

## 🚀 Overview

This project uses a **Voting Classifier (Decision Tree + Random Forest + XGBoost)** trained on a symptom-based dataset to predict diseases. The web interface allows users to input symptoms through a dropdown or text input and receive:

* Predicted disease
* Confidence score
* Top 3 possible diseases
* Disease description
* Precautions
* Severity score
* SHAP-based feature importance (model explainability)

---

## ✨ Features

* ✅ Ensemble ML model (VotingClassifier)
* ✅ Symptom-based prediction
* ✅ Confidence score display
* ✅ Top 3 disease predictions
* ✅ SHAP explainability
* ✅ Disease descriptions & precautions
* ✅ Severity scoring system
* ✅ Clean Streamlit UI (no custom HTML)

---

## 🛠️ Tech Stack

* **Python 3.11 (Anaconda)**
* **Streamlit**
* **Scikit-learn**
* **XGBoost**
* **SHAP**
* **Pandas, NumPy**

----

## 📁 Project Structure

```
Disease_Pred/
│
├── files/
│   ├── train.py
│   ├── app.py
│   ├── predict.py
│   ├── model.pkl
│   ├── disease_dataset.csv.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   ├── Symptom-severity.csv
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/disease-prediction-app.git
cd disease-prediction-app/files
```

### 2. Create environment (optional but recommended)

```bash
conda create -n disease_pred python=3.11
conda activate disease_pred
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

*(If no requirements.txt, install manually: streamlit, sklearn, xgboost, shap, pandas, numpy)*

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

OR 

```bash
C:\Users\DELL\anaconda3\python.exe -m streamlit run app.py
```

---

## 🧠 Model Details

* **Algorithm:** Voting Classifier

  * Decision Tree
  * Random Forest
  * XGBoost
* **Training Data:** Symptom-based dataset (17 symptoms per disease)
* **Accuracy:** ~100% (on given dataset)

---

## 📊 Datasets Used

1. **disease_dataset.csv.csv**

   * Symptoms → Disease mapping

2. **symptom_Description.csv**

   * Disease descriptions

3. **symptom_precaution.csv**

   * Precautions for each disease

4. **Symptom-severity.csv**

   * Severity weights for symptoms

---

## 🔮 Future Improvements

* 🔹 Deploy on Streamlit Cloud / AWS
* 🔹 Add user authentication
* 🔹 Integrate chatbot for symptom guidance
* 🔹 Improve dataset (real-world medical data)
* 🔹 Add multilingual support
* 🔹 Mobile-friendly UI

---

## ⚠️ Disclaimer

This application is for **educational purposes only** and should not be used as a substitute for professional medical advice.

---

## 👨‍💻 Author

**Livas Dasari**

* GitHub: https://github.com/LeelaSrinivas-Dasari
* LinkedIn: www.linkedin.com/in/leelasrinivas-dasari

---
