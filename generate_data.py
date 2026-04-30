"""
generate_data.py
────────────────
Run this ONCE to create all three CSV files the Market Skill Gap Analyzer needs.

Usage:
    python generate_data.py

Output files (place in the same folder as app.py):
    job_skills_data.csv      – job postings with required skills & salaries
    courses_data.csv         – online courses mapped to skills
    market_trends_data.csv   – monthly demand index per skill (2020–2024)
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────
# 1.  MASTER SKILL & ROLE DEFINITIONS
# ──────────────────────────────────────────────────────────
ROLES = {
    "Data Scientist": {
        "core":     ["Python", "Machine Learning", "Statistics", "SQL", "Data Visualization"],
        "optional": ["Deep Learning", "NLP", "Spark", "R", "Tableau", "AWS"],
        "base_salary": 1_400_000,
    },
    "Data Analyst": {
        "core":     ["SQL", "Excel", "Data Visualization", "Statistics", "Python"],
        "optional": ["Tableau", "Power BI", "R", "Google Analytics"],
        "base_salary": 850_000,
    },
    "Machine Learning Engineer": {
        "core":     ["Python", "Machine Learning", "Deep Learning", "MLOps", "Docker"],
        "optional": ["Kubernetes", "AWS", "GCP", "Spark", "Scala"],
        "base_salary": 1_800_000,
    },
    "Backend Developer": {
        "core":     ["Python", "REST APIs", "SQL", "Docker", "Git"],
        "optional": ["Node.js", "Go", "Kubernetes", "PostgreSQL", "Redis", "AWS"],
        "base_salary": 1_200_000,
    },
    "Frontend Developer": {
        "core":     ["JavaScript", "React", "HTML/CSS", "TypeScript", "Git"],
        "optional": ["Vue.js", "Next.js", "Figma", "GraphQL", "Webpack"],
        "base_salary": 1_100_000,
    },
    "Full Stack Developer": {
        "core":     ["JavaScript", "React", "Node.js", "SQL", "Docker", "Git"],
        "optional": ["TypeScript", "AWS", "MongoDB", "GraphQL", "Next.js"],
        "base_salary": 1_350_000,
    },
    "DevOps Engineer": {
        "core":     ["Docker", "Kubernetes", "CI/CD", "Linux", "AWS"],
        "optional": ["Terraform", "Ansible", "Python", "GCP", "Azure", "Monitoring"],
        "base_salary": 1_500_000,
    },
    "Cloud Architect": {
        "core":     ["AWS", "Azure", "GCP", "Kubernetes", "Terraform", "Networking"],
        "optional": ["Docker", "CI/CD", "Security", "Cost Optimization", "Linux"],
        "base_salary": 2_200_000,
    },
    "Product Manager": {
        "core":     ["Product Strategy", "Agile", "Data Analysis", "Roadmapping", "Stakeholder Management"],
        "optional": ["SQL", "Jira", "A/B Testing", "User Research", "Figma"],
        "base_salary": 1_600_000,
    },
    "UI/UX Designer": {
        "core":     ["Figma", "User Research", "Wireframing", "Prototyping", "HTML/CSS"],
        "optional": ["Adobe XD", "Sketch", "Motion Design", "Usability Testing", "Accessibility"],
        "base_salary": 900_000,
    },
    "Cybersecurity Analyst": {
        "core":     ["Network Security", "Linux", "Python", "Penetration Testing", "SIEM"],
        "optional": ["AWS Security", "Ethical Hacking", "Incident Response", "Compliance", "Forensics"],
        "base_salary": 1_300_000,
    },
    "NLP Engineer": {
        "core":     ["Python", "NLP", "Deep Learning", "Transformers", "Machine Learning"],
        "optional": ["Hugging Face", "BERT", "GPT", "Spark", "AWS"],
        "base_salary": 1_700_000,
    },
}

COMPANIES   = ["Google", "Microsoft", "Amazon", "Flipkart", "Zomato", "Razorpay",
               "Paytm", "Infosys", "TCS", "Wipro", "Accenture", "Deloitte",
               "PhonePe", "Swiggy", "BYJU'S", "OLA", "Nykaa", "Meesho",
               "Freshworks", "Zoho", "Salesforce", "Adobe", "IBM", "Capgemini"]

LOCATIONS   = ["Bengaluru", "Hyderabad", "Mumbai", "Delhi NCR", "Pune",
               "Chennai", "Kolkata", "Ahmedabad", "Remote"]

EXPERIENCE  = ["0-2 yrs", "2-5 yrs", "5-8 yrs", "8+ yrs"]

EXP_MULT    = {"0-2 yrs": 0.75, "2-5 yrs": 1.0, "5-8 yrs": 1.35, "8+ yrs": 1.75}

# ──────────────────────────────────────────────────────────
# 2.  JOB POSTINGS  (job_skills_data.csv)
# ──────────────────────────────────────────────────────────
records = []
for _ in range(2000):
    role      = random.choice(list(ROLES.keys()))
    meta      = ROLES[role]
    exp       = random.choice(EXPERIENCE)
    company   = random.choice(COMPANIES)
    location  = random.choice(LOCATIONS)

    # pick 3-5 core + 0-3 optional skills
    n_core    = min(len(meta["core"]),    random.randint(3, 5))
    n_opt     = min(len(meta["optional"]), random.randint(0, 3))
    skills    = random.sample(meta["core"], n_core) + random.sample(meta["optional"], n_opt)

    salary    = meta["base_salary"] * EXP_MULT[exp] * random.uniform(0.85, 1.20)
    if location in ["Bengaluru", "Mumbai", "Delhi NCR"]:
        salary *= random.uniform(1.05, 1.15)

    records.append({
        "job_title":          role,
        "company":            company,
        "location":           location,
        "experience_required": exp,
        "required_skills":    "|".join(skills),
        "num_skills_required": len(skills),
        "salary_inr":         round(salary, -3),
        "posted_year":        random.choice([2022, 2023, 2024]),
        "remote":             "Yes" if location == "Remote" else random.choice(["Yes", "No"]),
    })

jobs_df = pd.DataFrame(records)
jobs_df.to_csv("job_skills_data.csv", index=False)
print(f"✅  job_skills_data.csv  → {len(jobs_df):,} rows")


# ──────────────────────────────────────────────────────────
# 3.  COURSES  (courses_data.csv)
# ──────────────────────────────────────────────────────────
COURSE_DB = [
    # Python
    ("Python for Everybody",              "Coursera",   "Python",            "Beginner",      1499,  4.8, "https://coursera.org"),
    ("Complete Python Bootcamp",          "Udemy",      "Python",            "Beginner",       499,  4.7, "https://udemy.com"),
    ("Python Data Science Handbook",      "O'Reilly",   "Python",            "Intermediate",  1999,  4.6, "https://oreilly.com"),
    # SQL
    ("SQL for Data Science",              "Coursera",   "SQL",               "Beginner",      1299,  4.7, "https://coursera.org"),
    ("The Complete SQL Bootcamp",         "Udemy",      "SQL",               "Beginner",       399,  4.8, "https://udemy.com"),
    # Machine Learning
    ("Machine Learning Specialization",   "Coursera",   "Machine Learning",  "Intermediate",  3999,  4.9, "https://coursera.org"),
    ("ML A-Z: Hands-On Python & R",      "Udemy",      "Machine Learning",  "Intermediate",   599,  4.6, "https://udemy.com"),
    # Deep Learning
    ("Deep Learning Specialization",      "Coursera",   "Deep Learning",     "Advanced",      4999,  4.9, "https://coursera.org"),
    ("Practical Deep Learning",           "fast.ai",    "Deep Learning",     "Advanced",         0,  4.8, "https://fast.ai"),
    # NLP
    ("NLP with Transformers",             "Hugging Face","NLP",              "Advanced",         0,  4.7, "https://huggingface.co"),
    ("NLP Specialization",                "Coursera",   "NLP",               "Intermediate",  3999,  4.8, "https://coursera.org"),
    ("Transformers for NLP",              "Udemy",      "Transformers",      "Advanced",       699,  4.6, "https://udemy.com"),
    # Data Visualization
    ("Data Visualization with Tableau",   "Coursera",   "Tableau",           "Beginner",      1799,  4.6, "https://coursera.org"),
    ("Data Visualization with Python",    "edX",        "Data Visualization","Intermediate",  2499,  4.5, "https://edx.org"),
    ("Power BI Desktop",                  "Udemy",      "Power BI",          "Beginner",       449,  4.7, "https://udemy.com"),
    # Cloud
    ("AWS Solutions Architect",           "A Cloud Guru","AWS",              "Intermediate",  2999,  4.8, "https://acloudguru.com"),
    ("GCP Professional Data Engineer",    "Google",     "GCP",               "Advanced",      3499,  4.7, "https://cloud.google.com"),
    ("Azure Fundamentals AZ-900",         "Microsoft",  "Azure",             "Beginner",      1999,  4.6, "https://learn.microsoft.com"),
    # Docker / Kubernetes
    ("Docker & Kubernetes: The Guide",    "Udemy",      "Docker",            "Intermediate",   599,  4.7, "https://udemy.com"),
    ("Kubernetes for Developers",         "Linux Foundation","Kubernetes",   "Advanced",      4999,  4.8, "https://training.linuxfoundation.org"),
    # DevOps
    ("DevOps Bootcamp",                   "Udemy",      "CI/CD",             "Intermediate",   699,  4.7, "https://udemy.com"),
    ("Terraform Associate",               "HashiCorp",  "Terraform",         "Intermediate",  2999,  4.6, "https://hashicorp.com"),
    # React / Frontend
    ("React — The Complete Guide",        "Udemy",      "React",             "Intermediate",   499,  4.8, "https://udemy.com"),
    ("JavaScript Algorithms",             "Udemy",      "JavaScript",        "Beginner",       399,  4.7, "https://udemy.com"),
    ("TypeScript Deep Dive",              "GitHub",     "TypeScript",        "Intermediate",     0,  4.6, "https://github.com"),
    # Design
    ("Figma UI/UX Design",               "Udemy",      "Figma",             "Beginner",       449,  4.8, "https://udemy.com"),
    ("User Research Methods",             "Interaction Design","User Research","Beginner",    1299,  4.5, "https://interaction-design.org"),
    # Statistics
    ("Statistics with Python",            "Coursera",   "Statistics",        "Intermediate",  1999,  4.6, "https://coursera.org"),
    ("Bayesian Statistics",               "edX",        "Statistics",        "Advanced",      2999,  4.5, "https://edx.org"),
    # Cybersecurity
    ("Google Cybersecurity Certificate",  "Coursera",   "Network Security",  "Beginner",      3999,  4.8, "https://coursera.org"),
    ("Ethical Hacking Bootcamp",          "Udemy",      "Ethical Hacking",   "Intermediate",   699,  4.7, "https://udemy.com"),
    ("SIEM & SOC Analyst Training",       "Cybrary",    "SIEM",              "Intermediate",  1999,  4.5, "https://cybrary.it"),
    # Product
    ("Product Management Fundamentals",   "Coursera",   "Product Strategy",  "Beginner",      2499,  4.6, "https://coursera.org"),
    ("Agile with Atlassian Jira",         "Coursera",   "Agile",             "Beginner",       999,  4.5, "https://coursera.org"),
    # Spark / Big Data
    ("Apache Spark with Python",          "Udemy",      "Spark",             "Intermediate",   599,  4.6, "https://udemy.com"),
    # MLOps
    ("MLOps Specialization",              "Coursera",   "MLOps",             "Advanced",      4999,  4.8, "https://coursera.org"),
    ("MLflow for MLOps",                  "Udemy",      "MLOps",             "Intermediate",   499,  4.5, "https://udemy.com"),
]

courses_df = pd.DataFrame(COURSE_DB, columns=[
    "course_name", "platform", "skill_covered", "level", "price_inr", "rating", "url"
])
courses_df.to_csv("courses_data.csv", index=False)
print(f"✅  courses_data.csv     → {len(courses_df):,} rows")


# ──────────────────────────────────────────────────────────
# 4.  MARKET TRENDS  (market_trends_data.csv)
# ──────────────────────────────────────────────────────────
SKILL_TRENDS = {
    # AI / ML boom
    "Python":           (72, 6, 0.4),
    "Machine Learning": (55, 8, 0.6),
    "Deep Learning":    (40, 9, 0.7),
    "NLP":              (30, 12, 0.9),
    "Transformers":     (15, 18, 1.2),
    "MLOps":            (20, 14, 1.0),
    # Cloud
    "AWS":              (65, 5, 0.3),
    "GCP":              (30, 7, 0.5),
    "Azure":            (42, 6, 0.4),
    "Kubernetes":       (35, 8, 0.6),
    "Docker":           (50, 5, 0.3),
    "Terraform":        (22, 10, 0.7),
    # Data
    "SQL":              (80, 2, 0.1),
    "Spark":            (38, 4, 0.2),
    "Tableau":          (42, 3, 0.1),
    "Power BI":         (38, 5, 0.3),
    # Web
    "React":            (55, 4, 0.2),
    "TypeScript":       (40, 7, 0.5),
    "JavaScript":       (75, 2, 0.1),
    "Node.js":          (48, 3, 0.2),
    # Other
    "Figma":            (35, 6, 0.4),
    "Cybersecurity":    (28, 9, 0.6),
    "Agile":            (60, 2, 0.1),
    "Statistics":       (45, 3, 0.2),
}

months = pd.date_range("2020-01", "2024-12", freq="MS")
trend_records = []
for skill, (base, monthly_growth, noise_scale) in SKILL_TRENDS.items():
    for i, month in enumerate(months):
        demand = base + monthly_growth * (i / 6) + np.random.normal(0, noise_scale * base * 0.05)
        trend_records.append({
            "month":        month.strftime("%Y-%m"),
            "skill":        skill,
            "demand_index": max(0, round(demand, 2)),
        })

trends_df = pd.DataFrame(trend_records)
trends_df.to_csv("market_trends_data.csv", index=False)
print(f"✅  market_trends_data.csv → {len(trends_df):,} rows")
print("\n🎉  All datasets ready. Place them in the same folder as app.py and run:  streamlit run app.py")
