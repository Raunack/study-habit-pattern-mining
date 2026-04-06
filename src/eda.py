import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from data_loader import load_data
from preprocessor import get_feature_lists

def generate_eda(df):
    """
    Generates and saves EDA plots.
    """
    if not os.path.exists("images"):
        os.makedirs("images")
        
    # 1. Distribution of Exam Scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['exam_score'], kde=True, bins=30)
    plt.title('Distribution of Exam Scores')
    plt.savefig('images/exam_score_dist.png')
    print("Saved images/exam_score_dist.png")
    plt.close()

    # 2. Correlation Heatmap (Numerical)
    numerical_features, _ = get_feature_lists()
    # Add target to correlation matrix
    if 'exam_score' not in numerical_features:
        numerical_features.append('exam_score')
        
    plt.figure(figsize=(12, 10))
    corr = df[numerical_features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('images/correlation_heatmap.png')
    print("Saved images/correlation_heatmap.png")
    plt.close()

    # 3. Study Hours vs Exam Score
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='study_hours_per_day', y='exam_score', data=df, alpha=0.5)
    plt.title('Study Hours vs Exam Score')
    plt.savefig('images/study_vs_score.png')
    print("Saved images/study_vs_score.png")
    plt.close()

    # 4. Social Media vs Exam Score
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='social_media_hours', y='exam_score', data=df, alpha=0.5)
    plt.title('Social Media Usage vs Exam Score')
    plt.savefig('images/social_vs_score.png')
    print("Saved images/social_vs_score.png")
    plt.close()
    
    # 5. Boxplot of Exam Score by Study Environment
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='study_environment', y='exam_score', data=df)
    plt.title('Exam Score by Study Environment')
    plt.xticks(rotation=45)
    plt.savefig('images/environment_vs_score.png')
    print("Saved images/environment_vs_score.png")
    plt.close()

if __name__ == "__main__":
    df = load_data("enhanced_student_habits_performance_dataset.csv")
    if df is not None:
        generate_eda(df)
