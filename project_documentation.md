
# Car Insurance Claim Prediction System 🚗

An AI-powered web application built with Streamlit and Machine Learning (Voting Classifier) to predict car insurance claims.

### How to Run:

1. Clone the repository and activate your virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

An end-to-end Machine Learning pipeline and interactive Streamlit web application designed to predict the likelihood of car insurance policyholders filing a claim. Built using modern data science methodologies, exploratory data analysis, feature engineering, ensemble modeling (Soft Voting Classifier), and a clean web interface.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Dataset Summary](#-dataset-summary)
- [Key Features &amp; Workflow](#-key-features--workflow)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Web Application (UI/UX)](#-web-application-uiux)
- [Installation &amp; Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)

---

## 📖 Overview

The primary objective of this project is to assist automotive insurance companies in assessing risk profiles for drivers and vehicles. By leveraging historical driving habits, personal demographics, and past violation records, the trained machine learning model predicts whether a customer will submit an insurance claim (`outcome = 1`) or not (`outcome = 0`).

### 🎯 Key Goals:

- Perform data preprocessing, missing value imputation, and categorical encoding.
- Engineer domain-specific interaction features to boost predictive performance.
- Compare baseline classifiers and optimize an ensemble **Voting Classifier** (Random Forest + Gradient Boosting).
- Serve the final model via a user-friendly, responsive **Streamlit** dashboard.

---

## 📁 Project Architecture

The repository adheres to standard production-ready machine learning directory structures:

```text
Car_Insurance_Project/
│
├── data/
│   ├── raw/
│   │   └── car_insurance.csv            # Raw dataset
│   └── processed/
│       └── car_insurance_encoded.csv    # Preprocessed & encoded dataset
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb       # Data exploration, cleaning, encoding
│   └── 02_model_training.ipynb          # Model comparison, tuning, and evaluation
│
├── models/
│   └── best_voting_model.pkl            # Serialized trained Voting Classifier
│
├── src/
│   ├── __init__.py                      # Package initialization marker
│   ├── preprocessing.py                 # Imputation & encoding module
│   ├── feature_engineering.py           # Advanced feature creation module
│   └── train.py                         # Model training and export script
│
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependency specifications
└── README.md                            # Comprehensive project documentation
```

---

## 📊 Dataset Summary

The dataset contains **10,000 driver records** across **18 features** detailing demographic traits, driving records, vehicle information, and insurance risk factors.

| Attribute               | Description                                                                            | Type               |
| :---------------------- | :------------------------------------------------------------------------------------- | :----------------- |
| `id`                  | Unique policyholder identifier                                                         | Numeric (Excluded) |
| `age`                 | Age category of the driver                                                             | Numeric            |
| `gender`              | Driver gender                                                                          | Binary             |
| `driving_experience`  | Driver experience tier (`0-9y`, `10-19y`, `20-29y`, `30y+`)                    | Categorical        |
| `education`           | Educational background (`high school`, `university`, `none`)                     | Categorical        |
| `income`              | Driver income tier (`upper class`, `middle class`, `working class`, `poverty`) | Categorical        |
| `credit_score`        | Calculated credit score (0.0 to 1.0)                                                   | Continuous         |
| `vehicle_ownership`   | Ownership status (1 = Owner, 0 = Non-owner)                                            | Binary             |
| `vehicle_year`        | Vehicle manufacturing tier (`before 2015`, `after 2015`)                           | Categorical        |
| `vehicle_type`        | Vehicle category (`sedan`, `sports car`)                                           | Categorical        |
| `annual_mileage`      | Total distance driven annually                                                         | Continuous         |
| `speeding_violations` | Number of recorded speeding violations                                                 | Numeric (Count)    |
| `duis`                | Number of driving under the influence incidents                                        | Numeric (Count)    |
| `past_accidents`      | Number of previous traffic accidents                                                   | Numeric (Count)    |
| `outcome`             | Target variable (1 = Claim filed, 0 = No claim)                                        | Binary Target      |

---

## 🛠️ Key Features & Workflow

### 1. Exploratory Data Analysis (EDA) & Cleaning

- **Missing Values:** Median imputation applied to `credit_score` (~9.8% missing) and `annual_mileage` (~9.5% missing) to prevent data loss.
- **Outliers & Ranges:** Inspected statistical bounds (`df.describe()`) to verify variable sanity. Extreme count values for violations/accidents were retained as legitimate high-risk indicators.
- **Categorical Encoding:** Converted non-numeric categorical attributes using **One-Hot Encoding** (`pd.get_dummies(drop_first=True)`).

### 2. Feature Engineering

Created domain-relevant interaction metrics to capture complex driver behavior:

1. **Accidents per Violation:** $\text{past\_accidents} / (\text{speeding\_violations} + 1)$
2. **Mileage-Violation Interaction:** $\text{annual\_mileage} \times (\text{speeding\_violations} + 1)$

---

## 🤖 Machine Learning Pipeline

### Model Benchmarking

Evaluated baseline performance across multiple supervised classification models using 80/20 train-test splitting with target stratification:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier** *(Top individual performer ~84% Accuracy)*

### Optimization & Ensemble Methods

- **Hyperparameter Tuning:** Optimized tree depth, learning rate, and estimators for Gradient Boosting using `GridSearchCV` (5-Fold Cross-Validation).
- **Ensemble Model:** Combined Random Forest and Gradient Boosting into a **Soft Voting Classifier**, achieving balanced predictions across both target classes with strong F1-scores.

### Feature Importance Summary

Model interpretability analysis reveals that **Driver Age**, **Vehicle Ownership**, **Credit Score**, and **Driving Experience** are the primary predictors determining insurance claim probabilities.

---

## 💻 Web Application (UI/UX)

The user interface is built with **Streamlit** following modern UI/UX standards:

- **Clean Light Theme:** Soft gray background (`#f8fafc`), dark slate typography (`#1e293b`), and vibrant blue call-to-action elements.
- **Structured Two-Column Input:** Form fields grouped cleanly into Driver Details and Vehicle/History metrics.
- **Real-Time Visual Indicators:** Displays an exact claim probability score, status delta metric, and risk progress bar.

---

## 🚀 Installation & Setup

### Prerequisites

- Python `3.9` or higher installed on your system.

### Step-by-Step Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/car-insurance-claim-prediction.git
   cd car-insurance-claim-prediction
   ```
2. **Create and Activate a Virtual Environment:**

   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Usage Guide

### 1. Run Data Preprocessing

Generate the clean, encoded dataset inside `data/processed/`:

```bash
python src/preprocessing.py
```

### 2. Run Feature Engineering & Model Training

Train the Soft Voting Classifier and export the model file (`models/best_voting_model.pkl`):

```bash
python src/train.py
```

### 3. Launch the Streamlit Web Application

Start the interactive application locally:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to test the system.

---

## 🧰 Technologies Used

- **Programming Language:** Python 3.9+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Pipeline:** Scikit-Learn, Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **IDE & Version Control:** VS Code, Git

---

## 🔮 Future Improvements

- [ ] Deploy the web app to **Streamlit Community Cloud** or **AWS EC2**.
- [ ] Add SHAP (SHapley Additive exPlanations) values to the web interface for model interpretability per user query.
- [ ] Incorporate advanced boosting algorithms like **XGBoost** and **LightGBM**.
- [ ] Implement an automated CI/CD pipeline using GitHub Actions.

```

---

### Summary of what was created:
I have generated a standard `README.md` file in the editor workspace containing:
1. **Clear Project Overview & Goals**
2. **Directory Architecture Diagram**
3. **Dataset Schema & Dictionary**
4. **Detailed Machine Learning Pipeline Explanation**
5. **Complete Setup & Installation Command Guide**
6. **Application Running Instructions**
```
