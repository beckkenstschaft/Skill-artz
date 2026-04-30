"""
Market Skill Gap Analyzer
─────────────────────────
Production-grade Streamlit dashboard for job seekers & recruiters.

Requirements:
    pip install streamlit pandas numpy plotly scikit-learn PyMuPDF

Data files needed (run generate_data.py first):
    job_skills_data.csv
    courses_data.csv
    market_trends_data.csv
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkillMap — Market Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&family=Epilogue:wght@300;400;500;600&display=swap');

:root {
  --bg:      #0a0c10;
  --surface: #111318;
  --card:    #161b24;
  --border:  #222a38;
  --mint:    #00e5b0;
  --mint2:   #80ffd8;
  --coral:   #ff6b6b;
  --amber:   #ffb347;
  --sky:     #4fc3f7;
  --lilac:   #b39ddb;
  --text:    #e8edf5;
  --muted:   #6b7a95;
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Epilogue', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.6rem 2.2rem 2rem !important; max-width: 1700px !important; }

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.sb-logo {
  font-family: 'Clash Display', sans-serif;
  font-size: 1.5rem; font-weight: 700;
  color: var(--mint) !important;
  letter-spacing: -0.01em;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.2rem;
}
.sb-logo span { color: var(--text); opacity: 0.6; font-weight: 400; }
.sb-section {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted) !important;
  margin: 1.2rem 0 0.5rem;
}

[data-testid="stSidebar"] .stRadio label {
  font-size: 0.88rem !important; color: var(--muted) !important;
  padding: 0.25rem 0; transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--text) !important; }

.hero {
  background: linear-gradient(135deg, #0d1420 0%, #0a1628 55%, #0a0c10 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2.2rem 2.8rem 2rem;
  margin-bottom: 1.6rem;
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -80px; right: -60px;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(0,229,176,0.1) 0%, transparent 65%);
  pointer-events: none;
}
.hero::after {
  content: '';
  position: absolute; bottom: -50px; left: 20%;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(79,195,247,0.07) 0%, transparent 65%);
  pointer-events: none;
}
.hero-eyebrow {
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--mint);
  background: rgba(0,229,176,0.08); border: 1px solid rgba(0,229,176,0.2);
  border-radius: 20px; padding: 0.22rem 0.9rem;
  display: inline-block; margin-bottom: 0.9rem;
}
.hero-title {
  font-family: 'Clash Display', sans-serif;
  font-size: 2.6rem; font-weight: 700;
  color: var(--text); letter-spacing: -0.03em;
  line-height: 1.1; margin-bottom: 0.5rem;
}
.hero-title .hi { color: var(--mint); }
.hero-sub { font-size: 0.95rem; color: var(--muted); font-weight: 300; max-width: 580px; }

.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
.kpi-card {
  flex: 1; min-width: 160px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px; padding: 1.3rem 1.5rem;
  position: relative; overflow: hidden;
  transition: transform 0.2s, border-color 0.25s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: var(--mint); }
.kpi-card::after {
  content: ''; position: absolute;
  bottom: 0; left: 0; right: 0; height: 3px; border-radius: 0 0 14px 14px;
}
.kpi-card.mint::after  { background: linear-gradient(90deg, var(--mint), var(--mint2)); }
.kpi-card.sky::after   { background: linear-gradient(90deg, var(--sky), #a0e8ff); }
.kpi-card.amber::after { background: linear-gradient(90deg, var(--amber), #ffe4a0); }
.kpi-card.coral::after { background: linear-gradient(90deg, var(--coral), #ffb0b0); }
.kpi-card.lilac::after { background: linear-gradient(90deg, var(--lilac), #e1d5ff); }
.kpi-label { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.45rem; font-weight: 600; }
.kpi-value { font-family: 'Clash Display', sans-serif; font-size: 1.9rem; font-weight: 600; color: var(--text); line-height: 1; }
.kpi-sub   { font-size: 0.76rem; color: var(--muted); margin-top: 0.35rem; }
.kpi-icon  { position: absolute; top: 1rem; right: 1.1rem; font-size: 1.4rem; opacity: 0.15; }

.sh {
  font-family: 'Clash Display', sans-serif;
  font-size: 1.2rem; font-weight: 600; color: var(--text);
  margin: 0.5rem 0 1.1rem; display: flex; align-items: center; gap: 0.6rem;
}
.sh-line { flex: 1; height: 1px; background: var(--border); }

.tag-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.5rem; }
.tag {
  font-size: 0.78rem; font-weight: 500; border-radius: 20px;
  padding: 0.25rem 0.85rem; border: 1px solid;
  transition: transform 0.15s;
}
.tag:hover { transform: scale(1.04); }
.tag-have    { background: rgba(0,229,176,0.1);  border-color: rgba(0,229,176,0.35);  color: var(--mint); }
.tag-missing { background: rgba(255,107,107,0.1); border-color: rgba(255,107,107,0.35); color: var(--coral); }
.tag-neutral { background: rgba(107,122,149,0.12); border-color: rgba(107,122,149,0.3); color: var(--muted); }

.course-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
.course-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.3rem 1.4rem;
  transition: border-color 0.2s, transform 0.2s;
}
.course-card:hover { border-color: var(--mint); transform: translateY(-2px); }
.course-title { font-family: 'Clash Display', sans-serif; font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 0.5rem; line-height: 1.3; }
.course-meta  { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.7rem; }
.course-skill { font-size: 0.72rem; background: rgba(0,229,176,0.08); border: 1px solid rgba(0,229,176,0.2); color: var(--mint); border-radius: 12px; padding: 0.18rem 0.65rem; display: inline-block; margin-bottom: 0.6rem; }
.course-price { font-size: 0.82rem; font-weight: 600; color: var(--amber); }
.course-free  { font-size: 0.82rem; font-weight: 600; color: var(--mint); }
.course-stars { color: var(--amber); font-size: 0.8rem; }

.step-item {
  display: flex; align-items: flex-start; gap: 1rem;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 0.85rem 1.2rem;
  margin-bottom: 0.55rem; font-size: 0.87rem; color: var(--text);
  transition: border-color 0.2s;
}
.step-item:hover { border-color: var(--sky); }
.step-num { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; color: var(--mint); background: rgba(0,229,176,0.1); border-radius: 6px; padding: 0.2rem 0.5rem; white-space: nowrap; }

.pill-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.pill { background: rgba(79,195,247,0.08); border: 1px solid rgba(79,195,247,0.2); color: var(--sky); font-size: 0.76rem; font-weight: 500; border-radius: 20px; padding: 0.28rem 0.8rem; }

.stDataFrame, [data-testid="stDataFrame"] {
  background: var(--card) !important;
  border-radius: 12px !important; border: 1px solid var(--border) !important;
}
.sep { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.stSelectbox > div > div, .stMultiSelect > div > div {
  background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important;
}
.stTextArea textarea, .stTextInput input {
  background: var(--card) !important; border-color: var(--border) !important;
  color: var(--text) !important; border-radius: 10px !important;
}
.stFileUploader { background: var(--card) !important; border-color: var(--border) !important; border-radius: 12px !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: 10px; gap: 0.3rem; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; border-radius: 8px; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: var(--card) !important; color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Epilogue, sans-serif", color="#e8edf5", size=12),
    title_font=dict(family="Clash Display, sans-serif", size=14, color="#e8edf5"),
    xaxis=dict(gridcolor="#222a38", zeroline=False, tickfont=dict(color="#6b7a95"), title_font=dict(color="#6b7a95")),
    yaxis=dict(gridcolor="#222a38", zeroline=False, tickfont=dict(color="#6b7a95"), title_font=dict(color="#6b7a95")),
    legend=dict(bgcolor="rgba(22,27,36,0.8)", bordercolor="#222a38", borderwidth=1, font=dict(color="#e8edf5")),
    margin=dict(l=40, r=30, t=50, b=40),
    hoverlabel=dict(bgcolor="#161b24", bordercolor="#222a38", font=dict(color="#e8edf5", family="Epilogue"))
)
MINT_SCALE = [[0, "#0d1420"], [0.5, "#00a87f"], [1, "#00e5b0"]]
PALETTE    = ["#00e5b0", "#4fc3f7", "#ffb347", "#ff6b6b", "#b39ddb", "#80ffd8", "#ff8a65"]

def mc(fig):
    fig.update_layout(**PL)
    return fig

# ─────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_jobs():
    df = pd.read_csv("job_skills_data.csv")
    df["skills_list"] = df["required_skills"].apply(lambda x: str(x).split("|"))
    return df

@st.cache_data
def load_courses():
    return pd.read_csv("courses_data.csv")

@st.cache_data
def load_trends():
    df = pd.read_csv("market_trends_data.csv")
    df["month"] = pd.to_datetime(df["month"])
    return df

# ─────────────────────────────────────────────────────────────
# Salary model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_salary_model(jobs_df):
    mlb   = MultiLabelBinarizer()
    X     = mlb.fit_transform(jobs_df["skills_list"])
    exp_m = {"0-2 yrs": 1, "2-5 yrs": 3, "5-8 yrs": 6, "8+ yrs": 10}
    exp_e = jobs_df["experience_required"].map(exp_m).fillna(3).values.reshape(-1, 1)
    X_f   = np.hstack([X, exp_e])
    y     = jobs_df["salary_inr"].values
    Xt, Xe, yt, ye = train_test_split(X_f, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=10)
    model.fit(Xt, yt)
    return model, mlb, model.score(Xe, ye)

# ─────────────────────────────────────────────────────────────
# Skills master list
# ─────────────────────────────────────────────────────────────
ALL_KNOWN_SKILLS = [
    "Python","SQL","R","Java","Scala","Go","C++","JavaScript","TypeScript",
    "Machine Learning","Deep Learning","NLP","Statistics","MLOps","Transformers",
    "Docker","Kubernetes","AWS","GCP","Azure","Terraform","Ansible","CI/CD",
    "React","Node.js","Vue.js","Next.js","HTML/CSS","GraphQL",
    "Tableau","Power BI","Data Visualization","Excel","Spark",
    "Figma","Adobe XD","Sketch","Wireframing","Prototyping","User Research",
    "REST APIs","PostgreSQL","MongoDB","Redis","Git","Linux",
    "Agile","Jira","Product Strategy","Roadmapping","Stakeholder Management",
    "Network Security","Penetration Testing","SIEM","Ethical Hacking","Forensics",
    "A/B Testing","Google Analytics","Hugging Face","BERT","GPT",
]

def extract_skills_from_pdf(uploaded_file):
    if not PDF_SUPPORT:
        return []
    try:
        doc  = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = " ".join(p.get_text() for p in doc)
        return [s for s in ALL_KNOWN_SKILLS if re.search(re.escape(s), text, re.IGNORECASE)]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────
def compute_gap(user_skills, role, jobs_df):
    role_df  = jobs_df[jobs_df["job_title"] == role]
    all_req  = [s for sl in role_df["skills_list"] for s in sl]
    freq     = pd.Series(all_req).value_counts()
    required = set(freq[freq >= freq.quantile(0.35)].index.tolist())
    user_set = set(user_skills)
    matched  = user_set & required
    missing  = required - user_set
    extra    = user_set - required
    score    = int(round(len(matched) / len(required) * 100)) if required else 0
    return dict(required=required, matched=matched, missing=missing, extra=extra, score=score, freq=freq)

def predict_salary(user_skills, exp_label, model, mlb):
    exp_map = {"0-2 yrs": 1, "2-5 yrs": 3, "5-8 yrs": 6, "8+ yrs": 10}
    X = np.hstack([mlb.transform([user_skills]), np.array([[exp_map.get(exp_label, 3)]])])
    return max(0, model.predict(X)[0])

def score_color(s):
    return "#00e5b0" if s >= 75 else ("#ffb347" if s >= 50 else "#ff6b6b")

# ─────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────
def skill_freq_chart(freq, role):
    df = freq.head(12).reset_index()
    df.columns = ["Skill", "Postings"]
    fig = px.bar(df, x="Postings", y="Skill", orientation="h",
                 color="Postings", color_continuous_scale=MINT_SCALE,
                 title=f"Most Demanded Skills — {role}")
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    return mc(fig)

def gap_radar_chart(gap):
    skills = sorted(gap["required"])[:10]
    r_vals = [1 if s in gap["matched"] else 0 for s in skills]
    fig    = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r_vals, theta=skills, fill="toself",
        fillcolor="rgba(0,229,176,0.15)", line=dict(color="#00e5b0", width=2), name="Your Skills"))
    fig.add_trace(go.Scatterpolar(r=[1]*len(skills), theta=skills, fill="toself",
        fillcolor="rgba(79,195,247,0.06)", line=dict(color="#4fc3f7", width=1.5, dash="dot"), name="Required"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=False, range=[0, 1.2]),
            angularaxis=dict(tickfont=dict(color="#6b7a95", size=11), gridcolor="#222a38")),
        showlegend=True, title="Skill Coverage Radar",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf5"), title_font=dict(family="Clash Display,sans-serif", size=14),
        legend=dict(bgcolor="rgba(22,27,36,0.8)", bordercolor="#222a38", borderwidth=1),
        margin=dict(l=40, r=30, t=50, b=40),
    )
    return fig

def salary_benchmark_chart(role, salary_est, jobs_df):
    role_df = jobs_df[jobs_df["job_title"] == role]["salary_inr"]
    bins    = [0, 600000, 1000000, 1500000, 2000000, 3000000, 5000000]
    labels  = ["<6L","6–10L","10–15L","15–20L","20–30L","30L+"]
    binned  = pd.cut(role_df, bins=bins, labels=labels).value_counts().reindex(labels).reset_index()
    binned.columns = ["Range","Count"]
    fig = px.bar(binned, x="Range", y="Count", color="Count",
                 color_continuous_scale=MINT_SCALE, title=f"Salary Distribution — {role}")
    est_bin = pd.cut([salary_est], bins=bins, labels=labels)[0]
    if est_bin in labels:
        fig.add_vline(x=labels.index(est_bin), line_dash="dash", line_color="#ffb347",
                      line_width=2, annotation_text=f"Your Est: ₹{salary_est/100000:.1f}L",
                      annotation_font_color="#ffb347")
    fig.update_layout(coloraxis_showscale=False)
    return mc(fig)

def trends_chart(trends_df, skills):
    df  = trends_df[trends_df["skill"].isin(skills)]
    fig = px.line(df, x="month", y="demand_index", color="skill",
                  title="Market Demand Trend (2020–2024)", color_discrete_sequence=PALETTE)
    fig.update_traces(line_width=2.2)
    return mc(fig)

def skill_demand_heatmap(trends_df, skills):
    pivot = trends_df[trends_df["skill"].isin(skills)].pivot_table(
        index="skill", columns=trends_df["month"].dt.year,
        values="demand_index", aggfunc="mean").round(1)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=pivot.index.tolist(),
        colorscale=[[0,"#0d1420"],[0.5,"#006b52"],[1,"#00e5b0"]], hoverongaps=False))
    fig.update_layout(title="Annual Avg Demand by Skill",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf5"), title_font=dict(family="Clash Display,sans-serif", size=14),
        xaxis=dict(tickfont=dict(color="#6b7a95")), yaxis=dict(tickfont=dict(color="#6b7a95")),
        margin=dict(l=40, r=30, t=50, b=40))
    return fig

def recruiter_skill_chart(r_df, role):
    all_s = [s for sl in r_df["skills_list"] for s in sl]
    freq  = pd.Series(all_s).value_counts().head(14).reset_index()
    freq.columns = ["Skill","Demand Count"]
    fig = px.bar(freq, x="Skill", y="Demand Count", color="Demand Count",
                 color_continuous_scale=MINT_SCALE, title=f"Talent Demand Depth — {role}")
    fig.update_layout(coloraxis_showscale=False)
    return mc(fig)

def salary_by_exp_chart(jobs_df, role):
    order  = ["0-2 yrs","2-5 yrs","5-8 yrs","8+ yrs"]
    df     = jobs_df[jobs_df["job_title"] == role].groupby("experience_required", as_index=False)["salary_inr"].median()
    df["experience_required"] = pd.Categorical(df["experience_required"], categories=order, ordered=True)
    df     = df.sort_values("experience_required")
    df["salary_L"] = (df["salary_inr"] / 100000).round(1)
    fig = px.bar(df, x="experience_required", y="salary_L", color="salary_L",
                 color_continuous_scale=MINT_SCALE, title=f"Median Salary by Experience — {role}",
                 labels={"salary_L":"Salary (Lakhs)","experience_required":"Experience"})
    fig.update_layout(coloraxis_showscale=False)
    return mc(fig)

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
try:
    jobs_df    = load_jobs()
    courses_df = load_courses()
    trends_df  = load_trends()
    salary_model, salary_mlb, model_r2 = train_salary_model(jobs_df)
    DATA_LOADED = True
except FileNotFoundError as e:
    DATA_LOADED = False
    missing_file = str(e)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">Skill<span>Map</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["🎯 Job Seeker", "🏢 Recruiter / HR", "📈 Market Trends"], label_visibility="collapsed")
    st.markdown('<div class="sb-section">Quick Stats</div>', unsafe_allow_html=True)
    if DATA_LOADED:
        st.markdown(f'<p style="font-size:0.85rem;color:#e8edf5">📦 {jobs_df.shape[0]:,} job postings</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.85rem;color:#e8edf5">🎓 {courses_df.shape[0]} courses</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.85rem;color:#e8edf5">💼 {jobs_df["job_title"].nunique()} roles tracked</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.85rem;color:#e8edf5">🔧 {len(ALL_KNOWN_SKILLS)} skills indexed</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.85rem;color:#e8edf5">🤖 Model R²: {model_r2:.2f}</p>', unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#222a38;margin:1.4rem 0'>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.7rem;color:#6b7a95;">SkillMap Intelligence v1.0<br>Powered by ML · Skill Gap Engine</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────
if not DATA_LOADED:
    st.error(f"Data files not found. Run **generate_data.py** first.\n\n`{missing_file}`")
    st.code("python generate_data.py", language="bash")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🎯 AI-Powered Career Intelligence</div>
  <div class="hero-title">Market <span class="hi">Skill Gap</span> Analyzer</div>
  <div class="hero-sub">Pinpoint your skill gaps, predict your salary, and get tailored course recommendations — powered by live job market data.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Global KPIs
# ─────────────────────────────────────────────────────────────
avg_sal       = jobs_df["salary_inr"].mean()
median_sal    = jobs_df["salary_inr"].median()
top_role      = jobs_df.groupby("job_title")["salary_inr"].median().idxmax()
latest_month  = trends_df["month"].max()
hottest_skill = (trends_df[trends_df["month"] == latest_month]
                 .sort_values("demand_index", ascending=False).iloc[0]["skill"])

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card mint"><div class="kpi-icon">💼</div>
    <div class="kpi-label">Job Postings Analyzed</div>
    <div class="kpi-value">{jobs_df.shape[0]:,}</div>
    <div class="kpi-sub">Across {jobs_df['job_title'].nunique()} roles</div>
  </div>
  <div class="kpi-card sky"><div class="kpi-icon">💰</div>
    <div class="kpi-label">Avg. Market Salary</div>
    <div class="kpi-value">₹{avg_sal/100000:.1f}L</div>
    <div class="kpi-sub">Median ₹{median_sal/100000:.1f}L / yr</div>
  </div>
  <div class="kpi-card amber"><div class="kpi-icon">🏆</div>
    <div class="kpi-label">Highest Paying Role</div>
    <div class="kpi-value" style="font-size:1.05rem">{top_role}</div>
    <div class="kpi-sub">By median salary</div>
  </div>
  <div class="kpi-card coral"><div class="kpi-icon">🔥</div>
    <div class="kpi-label">Hottest Skill (2024)</div>
    <div class="kpi-value" style="font-size:1.35rem">{hottest_skill}</div>
    <div class="kpi-sub">Highest demand index</div>
  </div>
  <div class="kpi-card lilac"><div class="kpi-icon">📊</div>
    <div class="kpi-label">Skills Tracked</div>
    <div class="kpi-value">{len(ALL_KNOWN_SKILLS)}</div>
    <div class="kpi-sub">{courses_df['skill_covered'].nunique()} course categories</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# JOB SEEKER
# ═════════════════════════════════════════════════════════════
if mode == "🎯 Job Seeker":

    st.markdown('<div class="sh">🎯 Your Profile <span class="sh-line"></span></div>', unsafe_allow_html=True)

    c_input, c_config = st.columns([2, 1])

    with c_input:
        tab_manual, tab_pdf = st.tabs(["✏️  Enter Skills Manually", "📄  Upload Resume (PDF)"])
        with tab_manual:
            user_skills = st.multiselect(
                "Select your current skills",
                sorted(ALL_KNOWN_SKILLS),
                default=["Python", "SQL", "Machine Learning"],
                help="Pick every skill you are comfortable working with"
            )
        with tab_pdf:
            if PDF_SUPPORT:
                uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
                if uploaded:
                    extracted = extract_skills_from_pdf(uploaded)
                    if extracted:
                        st.success(f"✅ Extracted {len(extracted)} skills from your resume")
                        user_skills = extracted
                    else:
                        st.warning("No recognized skills found. Try the manual tab.")
                else:
                    st.info("Upload a PDF resume to auto-extract skills.")
            else:
                st.warning("PDF support requires: `pip install PyMuPDF`")

    with c_config:
        target_role = st.selectbox("🎯 Target Job Role", sorted(jobs_df["job_title"].unique()))
        exp_level   = st.selectbox("📅 Experience Level", ["0-2 yrs", "2-5 yrs", "5-8 yrs", "8+ yrs"])

    if not user_skills:
        st.info("👆 Select at least one skill to begin analysis.")
        st.stop()

    # ── Analysis ──
    gap     = compute_gap(user_skills, target_role, jobs_df)
    sal_est = predict_salary(user_skills, exp_level, salary_model, salary_mlb)
    col     = score_color(gap["score"])

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown('<div class="sh">📊 Gap Analysis Results <span class="sh-line"></span></div>', unsafe_allow_html=True)

    ka, kb, kc, kd = st.columns(4)
    with ka:
        st.markdown(f"""<div class="kpi-card mint">
          <div class="kpi-label">Match Score</div>
          <div class="kpi-value" style="color:{col}">{gap['score']}%</div>
          <div class="kpi-sub">vs {target_role}</div>
        </div>""", unsafe_allow_html=True)
    with kb:
        st.markdown(f"""<div class="kpi-card sky">
          <div class="kpi-label">Skills Matched</div>
          <div class="kpi-value">{len(gap['matched'])}</div>
          <div class="kpi-sub">of {len(gap['required'])} required</div>
        </div>""", unsafe_allow_html=True)
    with kc:
        st.markdown(f"""<div class="kpi-card coral">
          <div class="kpi-label">Skills Missing</div>
          <div class="kpi-value">{len(gap['missing'])}</div>
          <div class="kpi-sub">Need to acquire</div>
        </div>""", unsafe_allow_html=True)
    with kd:
        st.markdown(f"""<div class="kpi-card amber">
          <div class="kpi-label">Salary Estimate</div>
          <div class="kpi-value">₹{sal_est/100000:.1f}L</div>
          <div class="kpi-sub">{exp_level} · {target_role.split()[0]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Skill tags
    ts1, ts2 = st.columns(2)
    with ts1:
        st.markdown("**✅ Skills You Have (matched + bonus)**")
        tags = "".join(f'<span class="tag tag-have">{s}</span>' for s in sorted(gap["matched"]))
        tags += "".join(f'<span class="tag tag-neutral">{s}</span>' for s in sorted(gap["extra"]))
        st.markdown(f'<div class="tag-row">{tags or "<span style=color:#6b7a95>None matched yet</span>"}</div>', unsafe_allow_html=True)
    with ts2:
        st.markdown("**❌ Skills You Are Missing**")
        tags = "".join(f'<span class="tag tag-missing">{s}</span>' for s in sorted(gap["missing"]))
        st.markdown(f'<div class="tag-row">{tags or "<span style=color:#00e5b0>🎉 Fully qualified!</span>"}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(gap_radar_chart(gap), use_container_width=True)
    with ch2:
        st.plotly_chart(skill_freq_chart(gap["freq"], target_role), use_container_width=True)

    st.plotly_chart(salary_benchmark_chart(target_role, sal_est, jobs_df), use_container_width=True)

    # ── Course recommendations ──
    if gap["missing"]:
        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        st.markdown('<div class="sh">🎓 Recommended Courses for Your Gaps <span class="sh-line"></span></div>', unsafe_allow_html=True)

        rec = courses_df[courses_df["skill_covered"].isin(gap["missing"])].sort_values("rating", ascending=False)
        if rec.empty:
            rec = courses_df[courses_df["skill_covered"].str.lower().isin([s.lower() for s in gap["missing"]])]

        if not rec.empty:
            cards = '<div class="course-grid">'
            for _, row in rec.head(9).iterrows():
                stars = "★" * int(row["rating"]) + "☆" * (5 - int(row["rating"]))
                price = (f'<span class="course-free">🆓 Free</span>' if row["price_inr"] == 0
                         else f'<span class="course-price">₹{int(row["price_inr"]):,}</span>')
                cards += f"""<div class="course-card">
                  <div class="course-skill">{row['skill_covered']}</div>
                  <div class="course-title">{row['course_name']}</div>
                  <div class="course-meta">{row['platform']} · {row['level']}</div>
                  <div class="course-stars">{stars} {row['rating']}</div><br>{price}
                </div>"""
            cards += "</div>"
            st.markdown(cards, unsafe_allow_html=True)
        else:
            st.info("No specific courses mapped — search Coursera or Udemy directly for these skills.")

    # ── Trend for user's skills ──
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown('<div class="sh">📈 Market Demand for Your Skills <span class="sh-line"></span></div>', unsafe_allow_html=True)
    tracked = [s for s in user_skills if s in trends_df["skill"].unique()]
    if tracked:
        st.plotly_chart(trends_chart(trends_df, tracked[:8]), use_container_width=True)
    else:
        st.info("No trend data for your current skill selection.")


# ═════════════════════════════════════════════════════════════
# RECRUITER / HR
# ═════════════════════════════════════════════════════════════
elif mode == "🏢 Recruiter / HR":

    st.markdown('<div class="sh">🏢 Talent Market Intelligence <span class="sh-line"></span></div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns([2, 1, 1])
    with rc1:
        r_role = st.selectbox("Role to Analyze", sorted(jobs_df["job_title"].unique()))
    with rc2:
        r_loc  = st.multiselect("Location", ["All"] + sorted(jobs_df["location"].unique()), default=["All"])
    with rc3:
        r_exp  = st.multiselect("Experience", ["All", "0-2 yrs", "2-5 yrs", "5-8 yrs", "8+ yrs"], default=["All"])

    r_df = jobs_df[jobs_df["job_title"] == r_role].copy()
    if "All" not in r_loc: r_df = r_df[r_df["location"].isin(r_loc)]
    if "All" not in r_exp: r_df = r_df[r_df["experience_required"].isin(r_exp)]

    st.markdown(f"""<div class="pill-row">
      <div class="pill">📦 {r_df.shape[0]:,} postings</div>
      <div class="pill">📍 {r_df['location'].nunique()} locations</div>
      <div class="pill">💰 Median ₹{r_df['salary_inr'].median()/100000:.1f}L</div>
      <div class="pill">💰 Max ₹{r_df['salary_inr'].max()/100000:.1f}L</div>
      <div class="pill">🏢 {r_df['company'].nunique()} companies hiring</div>
    </div>""", unsafe_allow_html=True)

    rka, rkb, rkc, rkd = st.columns(4)
    with rka:
        st.markdown(f"""<div class="kpi-card mint"><div class="kpi-label">Postings</div>
        <div class="kpi-value">{r_df.shape[0]:,}</div><div class="kpi-sub">{r_role}</div></div>""", unsafe_allow_html=True)
    with rkb:
        st.markdown(f"""<div class="kpi-card sky"><div class="kpi-label">Avg Salary</div>
        <div class="kpi-value">₹{r_df['salary_inr'].mean()/100000:.1f}L</div><div class="kpi-sub">per annum</div></div>""", unsafe_allow_html=True)
    with rkc:
        top_loc = r_df["location"].value_counts().index[0] if not r_df.empty else "—"
        st.markdown(f"""<div class="kpi-card amber"><div class="kpi-label">Top City</div>
        <div class="kpi-value" style="font-size:1.1rem">{top_loc}</div><div class="kpi-sub">Most postings</div></div>""", unsafe_allow_html=True)
    with rkd:
        remote_pct = (r_df["remote"] == "Yes").mean() * 100
        st.markdown(f"""<div class="kpi-card coral"><div class="kpi-label">Remote Friendly</div>
        <div class="kpi-value">{remote_pct:.0f}%</div><div class="kpi-sub">of postings</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    rch1, rch2 = st.columns(2)
    with rch1:
        st.plotly_chart(recruiter_skill_chart(r_df if not r_df.empty else jobs_df, r_role), use_container_width=True)
    with rch2:
        st.plotly_chart(salary_by_exp_chart(jobs_df, r_role), use_container_width=True)

    # Salary by city
    loc_sal = (jobs_df[jobs_df["job_title"] == r_role]
               .groupby("location")["salary_inr"].median().reset_index())
    loc_sal["salary_L"] = (loc_sal["salary_inr"] / 100000).round(1)
    fig_loc = px.bar(loc_sal.sort_values("salary_L", ascending=False),
                     x="location", y="salary_L", color="salary_L",
                     color_continuous_scale=MINT_SCALE,
                     title=f"Median Salary by Location — {r_role}",
                     labels={"salary_L": "Salary (Lakhs)", "location": "City"})
    fig_loc.update_layout(coloraxis_showscale=False)
    st.plotly_chart(mc(fig_loc), use_container_width=True)

    # Companies
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown('<div class="sh">🏢 Companies Actively Hiring <span class="sh-line"></span></div>', unsafe_allow_html=True)
    co_df = (r_df.groupby("company").agg(
        postings=("job_title","count"),
        avg_salary=("salary_inr", lambda x: f"₹{x.mean()/100000:.1f}L")
    ).reset_index().sort_values("postings", ascending=False).head(15))
    st.dataframe(co_df, use_container_width=True, hide_index=True)

    # Candidate screener
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown('<div class="sh">🔍 Screen a Candidate <span class="sh-line"></span></div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        cand_skills = st.multiselect("Candidate's Skills", sorted(ALL_KNOWN_SKILLS), key="cand")
    with sc2:
        cand_exp = st.selectbox("Experience", ["0-2 yrs","2-5 yrs","5-8 yrs","8+ yrs"], key="cexp")

    if cand_skills:
        cg    = compute_gap(cand_skills, r_role, jobs_df)
        c_sal = predict_salary(cand_skills, cand_exp, salary_model, salary_mlb)
        cg1, cg2, cg3 = st.columns(3)
        c_col = score_color(cg["score"])
        verdict = ("✅ Strong Fit" if cg["score"] >= 75 else ("⚠️ Partial Fit" if cg["score"] >= 45 else "❌ Not Ready"))
        v_col   = "#00e5b0" if cg["score"] >= 75 else ("#ffb347" if cg["score"] >= 45 else "#ff6b6b")
        with cg1:
            st.markdown(f"""<div class="kpi-card mint"><div class="kpi-label">Match Score</div>
            <div class="kpi-value" style="color:{c_col}">{cg['score']}%</div>
            <div class="kpi-sub">for {r_role}</div></div>""", unsafe_allow_html=True)
        with cg2:
            st.markdown(f"""<div class="kpi-card sky"><div class="kpi-label">Market Salary Est.</div>
            <div class="kpi-value">₹{c_sal/100000:.1f}L</div><div class="kpi-sub">{cand_exp}</div></div>""", unsafe_allow_html=True)
        with cg3:
            st.markdown(f"""<div class="kpi-card amber"><div class="kpi-label">Verdict</div>
            <div class="kpi-value" style="color:{v_col};font-size:1rem">{verdict}</div>
            <div class="kpi-sub">Skill match basis</div></div>""", unsafe_allow_html=True)
        if cg["missing"]:
            st.markdown("**Skills to develop:**")
            tags = "".join(f'<span class="tag tag-missing">{s}</span>' for s in sorted(cg["missing"]))
            st.markdown(f'<div class="tag-row">{tags}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# MARKET TRENDS
# ═════════════════════════════════════════════════════════════
elif mode == "📈 Market Trends":

    st.markdown('<div class="sh">📈 Skill Demand Trends (2020–2024) <span class="sh-line"></span></div>', unsafe_allow_html=True)

    all_tracked = sorted(trends_df["skill"].unique().tolist())

    tc1, tc2 = st.columns([3, 1])
    with tc1:
        sel_skills = st.multiselect("Skills to compare", all_tracked,
            default=["Python","Machine Learning","Deep Learning","NLP","Transformers"])
    with tc2:
        view_type = st.selectbox("View as", ["Line Chart","Heatmap"])

    if sel_skills:
        if view_type == "Line Chart":
            st.plotly_chart(trends_chart(trends_df, sel_skills), use_container_width=True)
        else:
            st.plotly_chart(skill_demand_heatmap(trends_df, sel_skills), use_container_width=True)

    # Growth ranking
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown('<div class="sh">🚀 Fastest Growing Skills (2020 → 2024) <span class="sh-line"></span></div>', unsafe_allow_html=True)

    y2024 = trends_df[trends_df["month"].dt.year == 2024].groupby("skill")["demand_index"].mean()
    y2020 = trends_df[trends_df["month"].dt.year == 2020].groupby("skill")["demand_index"].mean()
    growth = ((y2024 - y2020) / y2020 * 100).dropna().sort_values(ascending=False).reset_index()
    growth.columns = ["Skill", "Growth (%)"]
    growth["Growth (%)"] = growth["Growth (%)"].round(1)
    growth["Trend"] = growth["Growth (%)"].apply(lambda x: "🚀 Surging" if x > 100 else ("📈 Growing" if x > 30 else "➡️ Stable"))

    ga, gb = st.columns([3, 2])
    with ga:
        fig_g = px.bar(growth.head(15), x="Growth (%)", y="Skill", orientation="h",
                       color="Growth (%)", color_continuous_scale=MINT_SCALE,
                       title="YoY Demand Growth 2020 → 2024")
        fig_g.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(mc(fig_g), use_container_width=True)
    with gb:
        st.markdown("**Growth Rankings**")
        st.dataframe(growth.head(15), use_container_width=True, hide_index=True)

    # Deep dive
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown('<div class="sh">🔬 Deep Dive — Single Skill <span class="sh-line"></span></div>', unsafe_allow_html=True)

    focus = st.selectbox("Pick a skill", all_tracked)
    fdf   = trends_df[trends_df["skill"] == focus].sort_values("month")

    fd1, fd2 = st.columns([3, 1])
    with fd1:
        fig_f = px.area(fdf, x="month", y="demand_index",
                        title=f"{focus} — Demand Index Over Time",
                        color_discrete_sequence=["#00e5b0"])
        fig_f.update_traces(fillcolor="rgba(0,229,176,0.1)", line=dict(color="#00e5b0", width=2.5))
        st.plotly_chart(mc(fig_f), use_container_width=True)
    with fd2:
        latest_v = fdf["demand_index"].iloc[-1]
        oldest_v = fdf["demand_index"].iloc[0]
        g_pct    = (latest_v - oldest_v) / oldest_v * 100
        g_color  = "#00e5b0" if g_pct > 0 else "#ff6b6b"
        st.markdown(f"""
        <div class="kpi-card mint" style="margin-bottom:1rem">
          <div class="kpi-label">2024 Demand Index</div>
          <div class="kpi-value">{latest_v:.1f}</div>
        </div>
        <div class="kpi-card coral">
          <div class="kpi-label">4-Year Growth</div>
          <div class="kpi-value" style="color:{g_color}">{g_pct:+.1f}%</div>
          <div class="kpi-sub">2020 → 2024</div>
        </div>""", unsafe_allow_html=True)

        rel = courses_df[courses_df["skill_covered"].str.lower() == focus.lower()]
        if not rel.empty:
            st.markdown("**Top Courses**")
            for _, row in rel.head(3).iterrows():
                price = "Free" if row["price_inr"] == 0 else f"₹{int(row['price_inr']):,}"
                st.markdown(f"""<div class="step-item">
                  <span class="step-num">{row['platform']}</span>
                  {row['course_name']} · ⭐{row['rating']} · {price}
                </div>""", unsafe_allow_html=True)
