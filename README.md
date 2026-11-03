
# 🚗 Car Insurance Claim Prediction System

An end-to-end Machine Learning web application designed to predict the likelihood of car insurance claims based on driver behavior and vehicle specifications. Built with Python, Scikit-Learn, and Streamlit.

---

## 📂 Project Structure

```text
Car_Insurance3/
│
├── data/
│   ├── raw/                  # Original raw dataset
│   └── processed/            # Cleaned and encoded datasets
│
├── models/
│   └── best_voting_model.pkl # Trained Machine Learning model
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb # Exploratory Data Analysis & Cleaning
│   └── 02_model_training.ipynb    # Model building & evaluation
│
├── src/
│   ├── __init__.py           # Package initializer
│   ├── preprocessing.py      # Data cleaning scripts
│   ├── feature_engineering.py# Feature engineering pipeline
│   └── train.py              # Model training script
│
├── app.py                    # Streamlit web application interface
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```
