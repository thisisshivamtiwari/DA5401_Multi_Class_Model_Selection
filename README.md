# 🛰️ DA5401 – Assignment 7  
## Multi-Class Model Selection using ROC & Precision–Recall Curves

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-green.svg)
![Grade](https://img.shields.io/badge/Grade-100%25-brightgreen.svg)

</div>

---

### 📌 Course: *Advanced Data Analytics (DA5401)*  
### 👤 Student: **Shivam Tiwari (DA25C019), IIT Madras**

---

## 📌 Project Overview

This assignment focuses on **multi-class model evaluation and selection** using advanced performance metrics beyond accuracy.  
Nine machine learning models (6 required + 3 bonus) are evaluated using:

- ✅ **ROC-AUC (One-vs-Rest)**
- ✅ **Precision–Recall Curve (PRC) and Average Precision**
- ✅ **Macro F1-Score**

Dataset used: **UCI Landsat Satellite (6 land cover classes, 36 spectral band features).**

> 🎯 Goal: Identify the **best** model for multi-class land cover classification.

---

## 🗂 Project Structure
├── DA5401_Assignment_7_Multi_Class_Model_Selection.ipynb   # Main notebook
├── sat.trn                                                 # Training dataset
├── sat.tst                                                 # Testing dataset
├── DA5401_A7_Assignment.pdf                                # Problem statement
└── README.md                                               # Documentation


---

## 📊 Models Compared

| Category | Models |
|----------|--------|
| **Required Models (6)** | KNN, Decision Tree, Dummy (Prior), Logistic Regression, Gaussian NB, SVC (probability=True) |
| **Bonus Models (3)** | Random Forest, XGBoost, Custom *BadClassifier* (inverted probabilities, AUC < 0.5) |

---

## ✅ Key Implementation Steps

### 1. Data Preparation
- Combined `sat.trn` + `sat.tst` → **6,435 samples**
- Applied **StandardScaler**
- Stratified **80/20** train-test split
- Encoded labels into `[0,1,2,3,4,5]` for XGBoost compatibility

### 2. Model Training
- Every model wrapped in **One-vs-Rest (OvR)** for multi-class ROC/PRC

### 3. Evaluation Metrics
- **ROC-AUC (Macro-Average)**
- **Average Precision (Macro-PRC-AP)**
- **Weighted Macro F1-Score**

---

## 🏆 Results

### ✅ Overall Performance

| Model | Macro F1 | ROC-AUC (Macro) | PRC-AP (Macro) |
|--------|---------|----------------|----------------|
| **XGBoost** | **0.9158** | **0.9926** | **0.9416** |
| **Random Forest** | 0.9079 | 0.9908 | 0.9388 |
| **SVC (linear)** | 0.8521 | 0.9634 | 0.8804 |
| Dummy (Prior) | 0.0543 | 0.5000 | 0.1670 |
| **BadClassifier (custom)** | 0.1002 | **0.0381** | 0.1343 |

### 🥇 Final Recommendation

> **XGBoost** is the best performing model — highest ROC-AUC, PRC-AP, and F1-Score.

---

## 🚀 How to Run

### Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

Launch notebook
jupyter notebook DA5401_Assignment_7_Multi_Class_Model_Selection.ipynb

Ensure sat.trn and sat.tst are in the same folder as the .ipynb.

Concepts Used
	•	One-vs-Rest (OvR): Convert multi-class problem into binary comparisons
	•	Macro-Averaging: Gives equal weight to each class
	•	Precision–Recall Curves: More useful than ROC when classes are imbalanced
	•	BadClassifier: Custom probability inversion to demonstrate AUC < 0.5 model

 
 Author

Shivam Tiwari
DA25C019 — IIT Madras


