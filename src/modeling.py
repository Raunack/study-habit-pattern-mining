import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from preprocessor import preprocess_data, get_feature_lists

def train_clustering_model(X, n_clusters=4):
    """
    Trains a K-Means clustering model to identify student segments.
    """
    print(f"\nTraining K-Means Clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Save model
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    print("✅ KMeans model saved to models/kmeans_model.pkl")
    
    # Evaluation (Silhouette Score on a sample to save time if dataset is huge)
    sample_size = min(10000, X.shape[0])
    score = silhouette_score(X.iloc[:sample_size], kmeans.labels_[:sample_size])
    print(f"Silhouette Score (on sample): {score:.4f}")
    
    return kmeans

def train_prediction_model(X, y):
    """
    Trains a Random Forest Regressor to predict exam scores.
    """
    print("\nTraining Random Forest Regressor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save model
    joblib.dump(rf, 'models/rf_model.pkl')
    print("✅ Random Forest model saved to models/rf_model.pkl")
    
    return rf

if __name__ == "__main__":
    from data_loader import load_data
    
    # Load and preprocess
    df = load_data("enhanced_student_habits_performance_dataset.csv")
    if df is not None:
        X, y = preprocess_data(df, save_pipeline=False) # Pipeline already saved
        
        # Train Clustering
        train_clustering_model(X, n_clusters=4)
        
        # Train Prediction
        train_prediction_model(X, y)
