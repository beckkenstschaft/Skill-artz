A.  SkillMap:

AI-powered Skill Gap Analyzer using Machine Learning that predicts salary, identifies missing skills, and recommends courses based on real-world job market data.


B. It allows users to:
- Identify missing skills for a target job role
- Estimate expected salary using machine learning
- Get course recommendations to close skill gaps
- Analyze market demand trends for skills

This project aids students and professionals in making informed career choices instead of guessing what to learn next.


C.  Key Features
- Skill Gap Analysis (Match % calculation)
- Salary Prediction using Ridge Regression
- Resume Skill Extraction (PDF parsing using PyMuPDF)
- Smart Course Recommendation System
- Interactive Dashboard (Streamlit + Plotly)
- Market Demand Trend Analysis (2020–2024)
- Recruiter Mode for talent analysis

D.  How It Works
1. User inputs skills manually or uploads a resume
2. System extracts and processes skills
3. Compares with job market data
4. Calculates:
      - Matched Skills
      - Missing Skills
      - Match Score (%)
5. Estimates salary using the ML model
6. Recommends courses based on missing skills
7. Displays insights with interactive visualizations


E.  Tech Stack
- Frontend: Streamlit
- Backend: Python
- Data Processing: Pandas, NumPy
- Visualization: Plotly
- Machine Learning: Scikit-learn (Ridge Regression)
- Resume Parsing: PyMuPDF


F.  Machine Learning Details
- Model: Ridge Regression
- Input Features:
        Skills (encoded using MultiLabelBinarizer)
        Experience Level
- Output:
       Salary Prediction

Ridge Regression is used to manage multicollinearity among skill features.


G.  Project Structure

market-skill-gap-analyzer/
│── skill_gap_app.py
│── generate_data.py
│── job_skills_data.csv
│── courses_data.csv
│── market_trends_data.csv
│── requirements.txt
│── README.md
│── .gitignore


H.  Getting Started

1. Clone the repository

2. Install dependencies
pip install -r requirements.txt

3. Generate dataset
python generate_data.py

4. Run the application
streamlit run skill_gap_app.py


I.  Use Cases
- Students planning career paths
- Job seekers preparing for roles
- Recruiters analyzing skill demand
- Data enthusiasts exploring market trends


J.  Future Improvements
- Integration with real-time job APIs (LinkedIn, Indeed)
- Advanced ML models (XGBoost, Neural Networks)
- Deployment on cloud (AWS / Streamlit Cloud)
- Mobile-friendly UI

Author:

Samradhi Gupta
