import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib

def get_feature_lists():
    """
    Returns lists of numerical and categorical features.
    """
    numerical_features = [
        'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 
        'attendance_percentage', 'sleep_hours', 'mental_health_rating', 
        'previous_gpa', 'stress_level', 'social_activity', 'screen_time',
        'motivation_level', 'exam_anxiety_score', 'time_management_score'
    ]
    
    categorical_features = [
        'gender', 'major', 'part_time_job', 'diet_quality', 
        'parental_education_level', 'internet_quality', 
        'extracurricular_participation', 'dropout_risk', 
        'study_environment', 'access_to_tutoring', 
        'family_income_range', 'parental_support_level', 'learning_style'
    ]
    
    return numerical_features, categorical_features

def preprocess_data(df, save_pipeline=True):
    """
    Preprocesses the dataframe:
    - Imputes missing values
    - Scales numerical features
    - Encodes categorical features
    """
    numerical_features, categorical_features = get_feature_lists()
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform
    print("Feature engineering...")
    X = df.drop(columns=['student_id', 'exam_score'], errors='ignore') # separate target
    y = df['exam_score'] if 'exam_score' in df.columns else None
    
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_names)
    
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    if save_pipeline:
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        print("✅ Preprocessor saved to models/preprocessor.pkl")
        
    return X_df, y

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data("enhanced_student_habits_performance_dataset.csv")
    if df is not None:
        X, y = preprocess_data(df)
        print("Data preprocessed successfully.")
        print(f"Processed shape: {X.shape}")
