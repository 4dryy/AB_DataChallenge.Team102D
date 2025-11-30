# 🚰 AB Data Challenge – Team 102.D  
### **Detecció de Consums Anòmals (Aigües de Barcelona)**

**Course:** Project Management (UPF – EUTOPIA Learning Unit)  
**Team:** 102.D  
**Academic Year:** 2025–2026  

---

## 🗂 Repository Structure

```
├── .vscode/
│
├── docs/                                 # PM Deliverables
│   ├── Team102D.Follow-upRegister.v0.1.pdf
│   ├── Team102D.ProjectCharter.v0.0.pdf
│   ├── Team102D.ProjectStakeholderMatrix.v0.1.pdf
│   └── Team102D.ProjectWorkPlan.v1.4.pdf
│
├── iteration_1/                           # Iteration 1 – Understanding & Planning
│   ├── 01_data_understanding.ipynb
│   └── 02_feature_plan.ipynb
│
├── iteration_2/                           # Iteration 2 – FE + Evaluation
│   ├── 01_iter2_data_analysis.ipynb
│   ├── 02_iter2_feature_engineering.ipynb
│   ├── 03_iter2_feature_evaluation.ipynb
│   ├── 04_iter2_prelim_dataset_prep.ipynb
│   └── 05_iter2_demonstration_increment.ipynb
│
├── iteration_3/                           # Iteration 3 – Final Pipeline
│   ├── config/
│   │   ├── params.json
│   │   └── paths.json
│   │
│   ├── notebooks/
│   │   ├── 01_iter3_data_cleaning.ipynb
│   │   ├── 02_iter3_feature_engineering.ipynb
│   │   ├── 03_iter3_feature_selection.ipynb
│   │   ├── 04_iter3_dataset_preparation.ipynb
│   │   ├── 05_iter3_model_training.ipynb
│   │   └── 06_iter3_product_pipeline_demo.ipynb
│   │
│   ├── results/
│   │   ├── models/
│   │   │   └── model_iter3_rf.joblib
│   │   ├── prepared/
│   │   │   ├── X_train.npy
│   │   │   ├── X_valid.npy
│   │   │   ├── X_test.npy
│   │   │   ├── y_train.npy
│   │   │   ├── y_valid.npy
│   │   │   └── y_test.npy
│   │   └── selected/
│   │       └── selected_features_iter3.txt
│
├── src/                                   # Python Package
│   ├── __init__.py
│   ├── cleaning.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── selection.py
│   ├── splitting.py
│   └── model_utils.py
│
├── .gitignore
└── README.md                              # This file
```

---

## ⚙️ Installation

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

If no requirements file is provided:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter
```

---

## 🚀 How to Run the Pipeline

### Option A – Notebook Workflow

Navigate to `iteration_3/notebooks/` and run notebooks in order:

1. 01_iter3_data_cleaning  
2. 02_iter3_feature_engineering  
3. 03_iter3_feature_selection  
4. 04_iter3_dataset_preparation  
5. 05_iter3_model_training  
6. 06_iter3_product_pipeline_demo  

Workflow:  
📥 ingest → 🧹 clean → 🧪 FE → 🔎 FS → 🗂 split → 🤖 train → 📊 demo

---

### Option B – Python Module Workflow

```python
from src.preprocessing import load_and_clean
from src.features import generate_features
from src.selection import select_features
from src.model_utils import train_isolation_forest

df = load_and_clean("path/to/input.parquet")
X = generate_features(df)
X_sel = select_features(X)
model = train_isolation_forest(X_sel)
```

---

## 🔍 Methodology Summary

### Iteration 1 – Foundations
- Data understanding  
- EDA planning  
- Feature plan  

### Iteration 2 – Development
- Feature engineering  
- Feature evaluation  
- Dataset preparation  
- Demo  

### Iteration 3 – Final Prototype
- Cleaning pipeline  
- Feature pipeline  
- Feature selection  
- Splits  
- Model training  
- Demo  

---

## 📊 Model Overview

- Algorithm: Isolation Forest  
- Type: Unsupervised anomaly detection  
- Justification: No labeled anomalies available  
- Outputs: anomaly score, binary flag  
- Model: `model_iter3_rf.joblib`

---

## 🧩 Key Features

- Dataset-agnostic pipeline  
- Modular Python package  
- Synthetic fallback dataset  
- Three-iteration agile workflow  
- Config-driven architecture  
- Full PM documentation  

---

## 👥 Team Members

- Adrià Cortés  
- Joan Company  
- Guillem García  
- Marc de Los Aires  
- Jofre Geli  

---

## 📄 License

Academic project for the UPF EUTOPIA Project Management course.

---

## 📬 Contact

Team 102.D – Universitat Pompeu Fabra
