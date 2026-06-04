# study-habit-pattern-mining

ML project that mines patterns from student study habit data to understand what behaviors actually correlate with academic performance. Includes a Streamlit dashboard for exploring the results.

---

## What it does

Takes a dataset of student habits — study hours, sleep, social media usage, attendance, etc. — and runs clustering and classification models to find patterns. The dashboard lets you explore the data visually, run predictions, and see which habits matter most.

---

## Stack

- Python
- Pandas, Scikit-learn
- Streamlit
- Dataset: `enhanced_student_habits_performance_dataset.csv` (included in repo)

---

## Running locally

```bash
git clone https://github.com/Raunack/study-habit-pattern-mining.git
cd study-habit-pattern-mining
pip install -r requirements.txt
streamlit run app.py
```

Visit http://localhost:8501

On Windows you can also just run `run_app.bat`.

---

## Structure

```
study-habit-pattern-mining/
├── app.py                  # Streamlit dashboard
├── src/                    # Data processing and model scripts
├── models/                 # Saved trained models
├── images/                 # Visualizations
├── enhanced_student_habits_performance_dataset.csv
└── requirements.txt
```

---
