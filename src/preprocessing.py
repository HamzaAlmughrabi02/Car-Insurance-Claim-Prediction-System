import os

import pandas as pd


def preprocess_data(file_path):
    # Read the raw dataset
    df = pd.read_csv(file_path)

    # 1. Handle missing values (avoiding inplace to prevent future warnings or errors)
    if 'credit_score' in df.columns:
        df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
    if 'annual_mileage' in df.columns:
        df['annual_mileage'] = df['annual_mileage'].fillna(df['annual_mileage'].median())

    # 2. Apply One-Hot Encoding to categorical columns
    categorical_cols = ['driving_experience', 'education', 'income', 'vehicle_year', 'vehicle_type']
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print("Data processing completed successfully. Final dataset shape:", df_encoded.shape)
    return df_encoded

if __name__ == "__main__":
    # Determine accurate file paths based on the 'src' directory location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(current_dir, "../data/raw/car_insurance.csv")
    processed_dir = os.path.join(current_dir, "../data/processed")

    # Ensure the target directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Execute preprocessing and save the processed file
    df_processed = preprocess_data(raw_path)
    output_file = os.path.join(processed_dir, "car_insurance_encoded.csv")
    df_processed.to_csv(output_file, index=False)

    print(f"✅ Processed file successfully saved at: {output_file}")