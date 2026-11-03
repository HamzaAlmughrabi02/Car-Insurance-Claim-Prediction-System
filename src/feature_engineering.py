import os
import pandas as pd

def add_advanced_features(df):
    df_adv = df.copy()
    
    # Create accidents per violation feature
    if 'past_accidents' in df_adv.columns and 'speeding_violations' in df_adv.columns:
        df_adv['accidents_per_violation'] = df_adv['past_accidents'] / (df_adv['speeding_violations'] + 1)
        
    # Create mileage and speeding violations interaction feature
    if 'annual_mileage' in df_adv.columns and 'speeding_violations' in df_adv.columns:
        df_adv['mileage_violation_interaction'] = df_adv['annual_mileage'] * (df_adv['speeding_violations'] + 1)
        
    print(f"✨ Advanced features successfully created. New shape: {df_adv.shape}")
    return df_adv

if __name__ == "__main__":
    # Determine the path and read the processed data to test feature engineering directly and print results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/processed/car_insurance_encoded.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df_featured = add_advanced_features(df)
        print("🚀 Feature engineering executed and result printed successfully!")
    else:
        print(f"⚠️ File not found at path: {data_path}. Please run preprocessing.py first.")