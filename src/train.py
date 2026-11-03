import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def train_and_save_model():
    # 1. Determine the current directory and load the preprocessed dataset we just created
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/processed/car_insurance_encoded.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found at path: {data_path}. Please run preprocessing.py first.")
        
    df_encoded = pd.read_csv(data_path)
    
    # 2. Separate features (X) from the target variable (y)
    X = df_encoded.drop(columns=['id', 'outcome'])
    y = df_encoded['outcome']
    
    # 3. Split data into 80% training and 20% testing while maintaining class balance (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Build the advanced Voting Classifier model (Random Forest + Gradient Boosting)
    print("🚀 Training the ensemble models (Voting Classifier)...")
    clf_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    
    voting_model = VotingClassifier(
        estimators=[('rf', clf_rf), ('gb', clf_gb)],
        voting='soft'
    )
    
    voting_model.fit(X_train, y_train)
    
    # 5. Evaluate the model performance
    y_pred = voting_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Training completed successfully! Model Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # 6. Save the final trained model in the 'models/' directory to be ready for Streamlit
    models_dir = os.path.join(current_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_voting_model.pkl")
    
    joblib.dump(voting_model, model_path)
    print(f"💾 Model successfully saved at path: {model_path}")

if __name__ == "__main__":
    train_and_save_model()