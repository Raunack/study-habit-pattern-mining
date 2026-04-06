import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(
    page_title="Student Study Habit Miner",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        kmeans = joblib.load('models/kmeans_model.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        return preprocessor, kmeans, rf_model
    except FileNotFoundError:
        st.error("Models not found. Please run src/modeling.py first.")
        return None, None, None

preprocessor, kmeans, rf_model = load_models()

def main():
    st.title("📚 Student Study Habit Pattern Miner")
    st.markdown("""
    This application analyzes your study habits to **predict your potential exam score** 
    and **identify your study pattern** (clustering).
    """)

    # Sidebar for inputs
    st.sidebar.header("📝 Enter Your Details")
    
    # Defaults based on dataset averages (approx)
    study_hours = st.sidebar.slider("Study Hours (Daily)", 0.0, 15.0, 3.5)
    sleep_hours = st.sidebar.slider("Sleep Hours (Daily)", 0.0, 12.0, 7.0)
    social_media = st.sidebar.slider("Social Media Hours", 0.0, 10.0, 2.0)
    attendance = st.sidebar.slider("Attendance (%)", 0.0, 100.0, 80.0)
    
    # Categorical Inputs
    major = st.sidebar.selectbox("Major", [
        'Business', 'Science', 'Arts', 'Engineering', 'Psychology', 
        'Computer Science', 'Biology', 'Other'
    ])
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
    part_time_job = st.sidebar.selectbox("Part-Time Job", ['Yes', 'No'])
    
    # Additional features required by the model
    # To keep the UI simple, we might use reasonable defaults for some, 
    # but let's expose the most important ones
    with st.sidebar.expander("More Details"):
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        motivation_level = st.slider("Motivation Level (1-10)", 1, 10, 5)
        diet_quality = st.selectbox("Diet Quality", ['Good', 'Fair', 'Poor'])
        internet_quality = st.selectbox("Internet Quality", ['High', 'Medium', 'Low'])
        study_env = st.selectbox("Study Environment", ['Library', 'Home', 'Cafe', 'Dorm', 'Quiet Room', 'Co-Learning Group'])
        parental_edu = st.selectbox("Parental Education", ['High School', 'Bachelor', 'Master', 'PhD', 'Some College'])
        learning_style = st.selectbox("Learning Style", ['Visual', 'Auditory', 'Kinesthetic', 'Reading'])
        
        # Hidden defaults for less critical inputs to avoid UI clutter
        # We construct the dataframe with ALL columns the preprocessor expects
        
    # Construct Input DataFrame
    input_data = {
        'age': [20], # Default
        'gender': [gender],
        'major': [major],
        'study_hours_per_day': [study_hours],
        'social_media_hours': [social_media],
        'netflix_hours': [0.5], # Default
        'part_time_job': [part_time_job],
        'attendance_percentage': [attendance],
        'sleep_hours': [sleep_hours],
        'diet_quality': [diet_quality],
        'exercise_frequency': [2], # Default
        'parental_education_level': [parental_edu],
        'internet_quality': [internet_quality],
        'mental_health_rating': [7], # Default
        'extracurricular_participation': ['No'], # Default
        'previous_gpa': [3.0], # Default
        'semester': [4], # Default
        'stress_level': [stress_level],
        'dropout_risk': ['No'], # Default
        'social_activity': [2], # Default
        'screen_time': [social_media + 2], # Approx
        'study_environment': [study_env],
        'access_to_tutoring': ['No'], # Default
        'family_income_range': ['Medium'], # Default
        'parental_support_level': ['Medium'], # Default
        'motivation_level': [motivation_level],
        'exam_anxiety_score': [5], # Default
        'learning_style': [learning_style],
        'time_management_score': [5] # Default
    }
    
    input_df = pd.DataFrame(input_data)
    
    if st.sidebar.button("Analyze My Habits"):
        if preprocessor and kmeans and rf_model:
            # Preprocess
            try:
                # We need to ensure the columns match the training data
                X_processed = preprocessor.transform(input_df)
                
                # Predict Score
                predicted_score = rf_model.predict(X_processed)[0]
                
                # Predict Cluster
                cluster_label = kmeans.predict(X_processed)[0]
                
                # Display Results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Predicted Exam Score")
                    st.metric(label="Score (0-100)", value=f"{predicted_score:.2f}")
                    
                    if predicted_score > 80:
                        st.success("Great job! You're on track for excellent results.")
                    elif predicted_score > 60:
                        st.warning("Good, but there's room for improvement.")
                    else:
                        st.error("Heads up! You might need to adjust your study habits.")
                        
                with col2:
                    st.subheader("🧬 Study Persona")
                    st.metric(label="Cluster ID", value=str(cluster_label))
                    st.info(f"You belong to Group {cluster_label}. This group typically has...")
                    # Interpretations would be hardcoded or derived here
                    
                # Visualizations
                st.subheader("📊 How You Compare")
                
                # Load a sample of data for comparison (just a quick hack to show distribution)
                # In production, we'd load grouped stats
                
                fig, ax = plt.figure(figsize=(10, 4)), plt.gca()
                # Create a simple gauge or bar chart
                comparison_data = pd.DataFrame({
                    'Metric': ['Study Hours', 'Sleep Hours', 'Social Media'],
                    'You': [study_hours, sleep_hours, social_media],
                    'Ideal (Approx)': [6.0, 8.0, 1.0] 
                })
                
                comparison_data = comparison_data.melt(id_vars='Metric', var_name='Type', value_name='Value')
                
                sns.barplot(x='Metric', y='Value', hue='Type', data=comparison_data, ax=ax)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                # st.write(input_df.columns) # For debugging
        else:
            st.error("Models failed to load.")

if __name__ == "__main__":
    main()
