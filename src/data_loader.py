import pandas as pd
import os

def load_data(filepath):
    """
    Loads the student dataset from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} was not found.")
    
    try:
        df = pd.read_csv(filepath)
        print("✅ Data loaded successfully.")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nData Types:")
        print(df.dtypes)
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Path is relative to where we run the script from the root of the project
    FILE_PATH = "enhanced_student_habits_performance_dataset.csv"
    load_data(FILE_PATH)
