import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

st.set_page_config(page_title="AI Electricity Theft Detection", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0d1117 0%, #0a0e14 100%); }
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #58a6ff 0%, #7ee787 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub { font-size: 1.1rem; color: #8b949e; margin-top: 0.3rem; margin-bottom: 1.5rem; }
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 22px 16px; text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    height: 100%;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #58a6ff; margin: 0; }
.metric-label {
    font-size: 0.78rem; color: #8b949e; text-transform: uppercase;
    letter-spacing: 1.5px; margin-top: 10px; font-weight: 600;
}
.info-card {
    background: #161b22; border: 1px solid #30363d;
    border-left: 4px solid #58a6ff; border-radius: 8px;
    padding: 18px 20px; height: 100%;
}
.info-card h3 { color: #58a6ff; margin: 0 0 8px 0; font-size: 0.9rem; letter-spacing: 1px; font-weight: 700; }
.info-card p { color: #c9d1d9; margin: 0; font-size: 0.92rem; line-height: 1.5; }
.step-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 22px 16px; text-align: center; height: 100%;
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 42px; height: 42px; border-radius: 50%;
    background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
    color: white; font-weight: 800; font-size: 1.1rem; margin-bottom: 12px;
}
.step-title { color: #e6edf3; font-weight: 700; font-size: 1rem; margin: 8px 0 6px 0; }
.step-desc { color: #8b949e; font-size: 0.83rem; }
.stButton>button {
    background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
    color: white; font-weight: 700; font-size: 1.05rem;
    padding: 14px 20px; border-radius: 10px; border: none;
    box-shadow: 0 4px 14px rgba(88,166,255,0.4); width: 100%;
}
.stButton>button:hover { box-shadow: 0 6px 20px rgba(88,166,255,0.6); color: white; }
.section-header {
    color: #e6edf3; font-size: 1.4rem; font-weight: 700;
    margin-top: 2rem; margin-bottom: 1rem;
    padding-bottom: 8px; border-bottom: 2px solid #21262d;
}
.badge {
    display: inline-block; padding: 4px 12px;
    background: rgba(88,166,255,0.15); color: #58a6ff;
    border: 1px solid rgba(88,166,255,0.4);
    border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 1px; margin-bottom: 12px;
}
.footer {
    text-align: center; color: #8b949e; padding: 30px 0 20px 0;
    font-size: 0.85rem; border-top: 1px solid #21262d; margin-top: 40px; line-height: 1.8;
}
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    n_estimators = st.slider("🌳 Number of Trees", 10, 200, 100, 10)
    sample_size = st.slider("👥 Sample Size (Users)", 5000, 20000, 15000, 1000)
    st.markdown("---")
    st.markdown("### 📌 Project Details")
    st.markdown("**Algorithm:** Random Forest  \n**Dataset:** SGCC (China)  \n**Records:** 42,372 users  \n**Days:** 1,035 per user")
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- Python 3.11\n- Scikit-learn\n- Streamlit\n- Pandas / NumPy\n- Matplotlib / Seaborn")

st.markdown('<span class="badge">🤖 AI-POWERED DETECTION SYSTEM</span>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">⚡ AI Electricity Theft Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Smart Grid Anomaly Detection using Random Forest Machine Learning</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="info-card"><h3>🎯 THE PROBLEM</h3>
    <p>Electricity theft costs utilities over <b>$96 billion/year</b> globally. Manual inspection is slow and expensive.</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="info-card"><h3>🤖 OUR SOLUTION</h3>
    <p>An AI model analyzes <b>daily consumption patterns</b> to automatically flag suspicious users for inspection.</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="info-card"><h3>📊 THE DATASET</h3>
    <p><b>SGCC Smart Grid Dataset</b> — 42,372 users × 1,035 days of real consumption data from China.</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_and_train(sample_size, n_estimators):
    DATA_PATH = '../data/sgcc_data.csv'
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    y = df.iloc[:, -1]
    X = df.iloc[:, 1:-1].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all').fillna(0).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'model': model, 'y_test': y_test, 'y_pred': y_pred,
        'accuracy': accuracy_score(y_test, y_pred),
        'X_shape': X.shape
    }

st.markdown('<h2 class="section-header">🚀 Run Live Detection</h2>', unsafe_allow_html=True)

if st.button("🎯  Train Model & Detect Theft"):
    with st.spinner("⏳ Training Random Forest... This takes 1-3 minutes."):
        result = load_and_train(sample_size, n_estimators)
    
    if result is None:
        st.error("❌ Dataset not found at `../data/sgcc_data.csv`")
    else:
        st.success("✅ Model trained successfully!")
        
        st.markdown('<h2 class="section-header">📊 Performance Metrics</h2>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{result["accuracy"]*100:.2f}%</p><p class="metric-label">Accuracy</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{result["X_shape"][0]:,}</p><p class="metric-label">Users Analyzed</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><p class="metric-value">{result["X_shape"][1]:,}</p><p class="metric-label">Day Features</p></div>', unsafe_allow_html=True)
        with m4:
            thefts = int(((result['y_test'] == 1) & (result['y_pred'] == 1)).sum())
            st.markdown(f'<div class="metric-card"><p class="metric-value">{thefts}</p><p class="metric-label">Thefts Detected</p></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">📈 Detailed Analysis</h2>', unsafe_allow_html=True)
        
        left, right = st.columns(2)
        with left:
            st.markdown("### 🎯 Confusion Matrix")
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#161b22')
            ax.set_facecolor('#161b22')
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Normal', 'Theft'], yticklabels=['Normal', 'Theft'],
                        ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            ax.set_ylabel('True Label', color='#e6edf3')
            ax.set_xlabel('Predicted Label', color='#e6edf3')
            ax.tick_params(colors='#e6edf3')
            plt.tight_layout()
            st.pyplot(fig)
        
        with right:
            st.markdown("### 📋 Classification Report")
            report = classification_report(result['y_test'], result['y_pred'],
                                           target_names=['Normal', 'Theft'], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True, height=340)
        
        st.markdown('<h2 class="section-header">🔍 Top Important Days</h2>', unsafe_allow_html=True)
        importances = result['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor('#161b22')
        ax2.set_facecolor('#161b22')
        ax2.bar(range(15), importances[indices], color='#58a6ff', edgecolor='#1f6feb')
        ax2.set_xticks(range(15))
        ax2.set_xticklabels([f"Day {i}" for i in indices], rotation=45, color='#e6edf3')
        ax2.set_ylabel('Importance', color='#e6edf3')
        ax2.tick_params(colors='#e6edf3')
        for spine in ['top', 'right']:
            ax2.spines[spine].set_visible(False)
        ax2.spines['bottom'].set_color('#30363d')
        ax2.spines['left'].set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig2)
        st.balloons()
else:
    st.info("👆 Click the **Train Model & Detect Theft** button above to run the AI.")

st.markdown('<h2 class="section-header">💡 How It Works</h2>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
steps = [
    ("1", "Load Data", "42,372 smart meter records from SGCC dataset"),
    ("2", "Clean & Prep", "Remove missing values, handle class imbalance"),
    ("3", "Train AI", "Random Forest learns from 100 decision trees"),
    ("4", "Detect Theft", "Flag suspicious users for utility inspection"),
]
for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
    with col:
        st.markdown(f'<div class="step-card"><div class="step-num">{num}</div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>', unsafe_allow_html=True)

st.markdown("""<div class="footer">
<b style="color:#58a6ff;">⚡ AI Electricity Theft Detection System</b><br>
ECE (AI & ML) Mini Project | 2026<br>
Built with Python • Scikit-learn • Streamlit
</div>""", unsafe_allow_html=True)