import os
import pickle

import streamlit as st

import db

st.set_page_config(page_title="NeuroGuard AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
db.init_db()

# ---------------------------------------------------------------------------
# REAL FEATURE SET — trained on the real "Stroke Risk Prediction Dataset
# Based on Symptoms" (Kaggle, mahatiratusher/stroke-risk-prediction-dataset).
# Order here MUST match FEATURE_COLUMNS in stroke_model.py exactly: age
# (scaled) first, then these 15 symptoms in this order.
# Note: this dataset has no gender column, so gender is recorded (asked in
# the form, saved to history) but NOT fed into the model — it was never
# trained on it.
# ---------------------------------------------------------------------------
SYMPTOMS = [
    ("chest_pain", "Chest pain"),
    ("shortness_of_breath", "Shortness of breath"),
    ("irregular_heartbeat", "Irregular heartbeat"),
    ("fatigue_weakness", "Fatigue or weakness"),
    ("dizziness", "Dizziness"),
    ("swelling_edema", "Swelling or edema"),
    ("neck_jaw_pain", "Pain in neck / jaw / shoulder / back"),
    ("excessive_sweating", "Excessive sweating"),
    ("persistent_cough", "Persistent cough"),
    ("nausea_vomiting", "Nausea or vomiting"),
    ("high_blood_pressure", "High blood pressure"),
    ("chest_discomfort", "Chest discomfort (during activity)"),
    ("cold_hands_feet", "Cold hands / feet"),
    ("snoring_sleep_apnea", "Snoring / sleep apnea"),
    ("anxiety_doom", "Anxiety / feeling of doom"),
]

# Static demo vitals shown on the Dashboard's 4 top cards — matches the
# inspo screenshot exactly. Your dataset/model doesn't track wearable
# metrics like heart rate/BP/sleep/steps, so these stay as illustrative
# sample data unless you wire up a real vitals source later.
DEMO_VITALS = [
    {"label": "Heart rate", "value": "72 bpm", "delta": "4%", "dir": "up", "icon": "❤️", "css": "vital-red"},
    {"label": "Blood pressure", "value": "128/82", "delta": "2%", "dir": "down", "icon": "💧", "css": "vital-green"},
    {"label": "Sleep", "value": "6.5 hrs", "delta": "8%", "dir": "up", "icon": "🌙", "css": "vital-amber"},
    {"label": "Steps", "value": "4,210", "delta": "12%", "dir": "up", "icon": "👣", "css": "vital-purple"},
]

# Sample risk panel shown before the user has run a real assessment —
# matches the inspo screenshot exactly (78% / Elevated / BP, Glucose, BMI).
SAMPLE_RISK = {
    "pct": 78, "label": "Elevated",
    "factors": [("Blood pressure", "High"), ("Glucose", "High"), ("BMI", "Elevated")],
    "recs": ["Exercise 30 min", "Reduce sodium", "Consult doctor", "Track weekly"],
}

RECOMMENDATION_RULES = {
    "high_blood_pressure": "Monitor blood pressure daily",
    "fatigue_weakness": "Prioritize rest and sleep",
    "shortness_of_breath": "Avoid strenuous exertion, consult doctor",
    "irregular_heartbeat": "Get an ECG checkup",
    "anxiety_doom": "Consider stress-reduction techniques",
}
DEFAULT_RECS = ["Consult a doctor", "Track weekly", "Stay hydrated", "Light daily exercise"]

APP_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(APP_DIR, "stroke_model.pkl")
    scaler_path = os.path.join(APP_DIR, "scaler.pkl")
    if not os.path.exists(model_path):
        return None, None, "missing_model"
    if not os.path.exists(scaler_path):
        return None, None, "missing_scaler"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, None


def run_prediction(age, symptom_values, model, scaler):
    age_scaled = scaler.transform([[age]])[0][0]
    input_data = [age_scaled] + symptom_values
    pred = model.predict([input_data])[0]
    try:
        proba = model.predict_proba([input_data])[0][1]  # P(at_risk)
    except Exception:
        proba = float(pred)
    return int(pred), round(proba * 100)


# ---------------------------------------------------------------------------
# STYLES  (dark NeuroGuard AI theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    :root{
        --navy-deep:#120c2b; --navy-card2:#1c1440; --accent:#f2a93b;
        --text-light:#f4f1fb; --text-dim:#b9b3d6;
    }
    .stApp { background:#f4f5fa; }
    header[data-testid="stHeader"] { background:#f4f5fa !important; }
    .block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1100px; }

    /* Streamlit fades each element in on mount; on a full-page rerun that
       can leave lower elements visibly translucent for a moment. Kill that
       specific transition so content renders at full opacity immediately. */
    [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"],
    [data-testid="stAppViewContainer"] [data-testid="element-container"],
    [data-testid="stAppViewContainer"] .element-container,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="element-container"] {
        transition: none !important; opacity: 1 !important;
    }

    section[data-testid="stSidebar"] { background:var(--navy-deep); border-right:1px solid #ffffff10; }
    section[data-testid="stSidebar"] * { color:var(--text-light) !important; }
    .brand { display:flex; align-items:center; gap:10px; padding:6px 4px 22px 4px; font-weight:800; font-size:1.15rem; }

    section[data-testid='stSidebar'] div[role='radiogroup'] { gap:6px; }
    section[data-testid='stSidebar'] div[role='radiogroup'] label {
        padding:11px 14px; border-radius:12px; width:100%; font-weight:500; color:var(--text-dim) !important;
    }
    section[data-testid='stSidebar'] div[role='radiogroup'] label:has(input:checked) {
        background: linear-gradient(90deg, #6a4bd6, #8a5cf0); box-shadow: 0 4px 14px #6a4bd655;
    }
    section[data-testid='stSidebar'] div[role='radiogroup'] label:has(input:checked) p { color:#fff !important; font-weight:600; }
    section[data-testid='stSidebar'] div[role='radiogroup'] input { display:none; }

    .hero { background: linear-gradient(135deg, var(--navy-card2), var(--navy-deep)); border-radius:22px;
            padding:30px 34px; position:relative; overflow:hidden; margin-bottom:22px; }
    .hero h1 { color:#fff; font-size:1.9rem; font-weight:800; margin:0 0 14px 0; }
    .pill { display:inline-flex; align-items:center; gap:6px; background:#ffffff14; color:#e8e4fa;
            padding:6px 14px; border-radius:999px; font-size:0.85rem; margin-right:10px; font-weight:500; }
    .hero-icon { position:absolute; right:34px; top:50%; transform:translateY(-50%); width:82px; height:82px;
                 border-radius:50%; background:#ffffff12; display:flex; align-items:center; justify-content:center; font-size:2.1rem; }
    .start-btn { display:inline-block; margin-top:20px; background:var(--accent); color:#241505; font-weight:700;
                 padding:12px 22px; border-radius:12px; font-size:0.95rem; box-shadow:0 6px 18px #f2a93b55;
                 text-decoration:none; }

    .vital-card { border-radius:18px; padding:20px 20px 16px 20px; height:118px; margin-bottom:20px;
                  display:flex; flex-direction:column; justify-content:space-between; }
    .vital-red{background:#3d1c1c;} .vital-green{background:#0f3a2c;} .vital-amber{background:#3a2c0e;} .vital-purple{background:#241a63;}
    .vital-top { display:flex; align-items:center; justify-content:space-between; }
    .vital-delta { font-size:0.8rem; font-weight:700; color:#8be3a8; }
    .vital-value { color:#fff; font-size:1.4rem; font-weight:800; margin-top:4px; }
    .vital-label { color:#d7d2ec; font-size:0.85rem; margin-top:2px; }
    .vital-bar { height:4px; border-radius:4px; background:#ffffff25; margin-top:8px; overflow:hidden; }
    .vital-bar span { display:block; height:100%; border-radius:4px; }
    .vital-red .vital-bar span { background:#ff8a7a; width:44%; }
    .vital-green .vital-bar span { background:#5fe3ac; width:72%; }
    .vital-amber .vital-bar span { background:#f7c04a; width:60%; }
    .vital-purple .vital-bar span { background:#b8a8ff; width:80%; }

    div.st-key-risk_gauge, div.st-key-risk_factors, div.st-key-risk_rec {
        background: linear-gradient(135deg, var(--navy-card2), var(--navy-deep)) !important;
        border-radius:22px !important; padding:26px 30px !important;
    }
    div.st-key-risk_gauge h3, div.st-key-risk_factors h3, div.st-key-risk_rec h3,
    div.st-key-risk_gauge h4, div.st-key-risk_factors h4, div.st-key-risk_rec h4 {
        color:var(--accent) !important; font-size:1rem !important; font-weight:700 !important; margin:0 0 14px 0 !important;
    }
    .factor-row { display:flex; justify-content:space-between; padding:9px 0; color:#e8e4fa; font-size:0.95rem; border-bottom:1px solid #ffffff10; }
    .factor-row .lvl-high { color:#ff8a7a; font-weight:700; }
    .rec-btn { display:block; text-align:center; background:#ffffff14; color:#e8e4fa; padding:11px 10px;
               border-radius:12px; font-size:0.88rem; font-weight:600; margin-bottom:10px; }
    .risk-label { color:var(--accent); font-weight:700; text-align:center; margin-top:8px; font-size:1.05rem; }

    .setup-warning { background:#3a2c0e; border-left:4px solid #f2a93b; padding:16px 20px; border-radius:8px; color:#f4f1fb; }

    div[data-testid="stMainBlockContainer"] div[data-testid="stButton"] button,
    div[data-testid="stMainBlockContainer"] div[data-testid="stFormSubmitButton"] button {
        background: var(--accent); color:#241505; font-weight:700; border:none;
        border-radius:12px; padding:10px 22px; box-shadow:0 6px 18px #f2a93b55;
    }
    div[data-testid="stMainBlockContainer"] div[data-testid="stButton"] button:hover,
    div[data-testid="stMainBlockContainer"] div[data-testid="stFormSubmitButton"] button:hover {
        background:#e39a2c; color:#241505;
    }

    .auth-wrap { max-width:420px; margin:60px auto 0 auto; }
    .auth-brand { text-align:center; color:#fff; font-weight:800; font-size:1.6rem; margin-bottom:28px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None  # dict: id, username, email

# ---------------------------------------------------------------------------
# AUTH GATE — sign up / log in before anything else renders
# ---------------------------------------------------------------------------
if st.session_state.auth_user is None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(160deg, #1c1440, #120c2b) !important; }
        header[data-testid="stHeader"] { background: linear-gradient(160deg, #1c1440, #120c2b) !important; }

        .auth-badge {
            width:64px; height:64px; margin:0 auto 18px auto; border-radius:50%;
            background: linear-gradient(135deg, #6a4bd6, #f2a93b);
            display:flex; align-items:center; justify-content:center; font-size:1.8rem;
            box-shadow: 0 0 0 6px #ffffff0d, 0 10px 30px #6a4bd655;
        }
        .auth-brand { text-align:center; color:#fff; font-weight:800; font-size:1.8rem; margin-bottom:6px;
                      letter-spacing:-0.02em; }
        .auth-tagline { text-align:center; color:#b9b3d6; font-size:0.92rem; margin-bottom:30px; }

        div.st-key-auth_card { max-width:420px; margin:0 auto; }
        div.st-key-auth_card div[data-testid="stTabs"] button[data-baseweb="tab"] { color:#8a84a8; font-weight:600; }
        div.st-key-auth_card div[data-testid="stTabs"] button[aria-selected="true"] { color:#f2a93b; }
        div.st-key-auth_card div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { background-color:#f2a93b; height:2.5px; }
        div.st-key-auth_card div[data-testid="stTabs"] div[data-baseweb="tab-border"] { background-color:#ffffff14; }
        div.st-key-auth_card div[data-testid="stForm"] {
            background: linear-gradient(160deg, #201959, #17102f);
            border:1px solid #ffffff14; border-radius:20px; padding:32px 30px;
            box-shadow: 0 20px 50px #00000040;
        }
        div.st-key-auth_card div[data-testid="stForm"] label p {
            color:#c9c3e8 !important; font-weight:500; font-size:0.88rem;
        }
        div.st-key-auth_card div[data-testid="stForm"] input {
            background:#120c2b !important; color:#f4f1fb !important;
            border:1px solid #ffffff1f !important; border-radius:10px !important; padding:10px 14px !important;
        }
        div.st-key-auth_card div[data-testid="stForm"] input:focus {
            border:1px solid #f2a93b !important; box-shadow:0 0 0 1px #f2a93b55 !important;
        }
        div.st-key-auth_card div[data-testid="stFormSubmitButton"] button { margin-top:8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="auth-wrap">
            <div class="auth-badge">🧠</div>
            <div class="auth-brand">NeuroGuard AI</div>
            <div class="auth-tagline">AI-powered stroke risk assessment &amp; prevention</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid, st.container(key="auth_card"):
        tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

        with tab_login:
            with st.form("login_form"):
                li_user = st.text_input("Username or email")
                li_pass = st.text_input("Password", type="password")
                li_submit = st.form_submit_button("Log in", use_container_width=True)
                if li_submit:
                    user = db.verify_user(li_user, li_pass)
                    if user:
                        st.session_state.auth_user = user
                        st.rerun()
                    else:
                        st.error("Incorrect username/email or password.")

        with tab_signup:
            with st.form("signup_form"):
                su_user = st.text_input("Username")
                su_email = st.text_input("Email")
                su_pass = st.text_input("Password", type="password")
                su_pass2 = st.text_input("Confirm password", type="password")
                su_submit = st.form_submit_button("Create account", use_container_width=True)
                if su_submit:
                    if su_pass != su_pass2:
                        st.error("Passwords don't match.")
                    else:
                        user, err = db.create_user(su_user, su_email, su_pass)
                        if err:
                            st.error(err)
                        else:
                            st.session_state.auth_user = user
                            st.rerun()
    st.stop()

user = st.session_state.auth_user

# ---------------------------------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------------------------------
NAV_OPTIONS = ["🖥️  Dashboard", "🩺  Risk check", "📰  News", "💬  Chatbot"]

# A keyed widget (key="nav_radio" below) ignores `index=` on every rerun
# after the first — it just keeps whatever the user last clicked. So to
# change pages from a button elsewhere (Start new assessment, Log out),
# we can't just write st.session_state.page; we have to write directly
# into st.session_state["nav_radio"] itself, and that write has to happen
# *before* st.radio(key="nav_radio") runs in this script pass. This flag
# is how other code "requests" a page change that takes effect next run.
if "pending_nav" in st.session_state:
    st.session_state["nav_radio"] = st.session_state.pop("pending_nav")

with st.sidebar:
    st.markdown('<div class="brand"><span>🧠</span> NeuroGuard AI</div>', unsafe_allow_html=True)
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = NAV_OPTIONS[0]
    st.session_state.page = st.radio(
        "nav", NAV_OPTIONS, label_visibility="collapsed", key="nav_radio",
    )
    st.markdown("---")
    st.caption(f"Logged in as **{user['username']}**")
    if st.button("Log out", key="logout_btn", use_container_width=True):
        st.session_state.auth_user = None
        st.session_state.pending_nav = NAV_OPTIONS[0]
        for k in ("messages", "chat_primed"):
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown("---")
    st.caption("Developed by Ashwin Muralidharan · MS Data Science, UNT")

model, scaler, model_error = load_model_and_scaler()

# ===========================================================================
# DASHBOARD
# ===========================================================================
if st.session_state.page == NAV_OPTIONS[0]:
    la = db.get_latest_assessment(user["id"])
    stats = db.get_stats(user["id"])

    top_l, top_r = st.columns([10, 1])
    with top_l:
        st.markdown('<div style="background:#fff;border-radius:14px;padding:12px 18px;color:#9b9bb0;border:1px solid #eceaf5;">🔍&nbsp;&nbsp;Search symptoms, reports...</div>', unsafe_allow_html=True)
    with top_r:
        st.markdown('<div style="background:#fff;border-radius:14px;width:44px;height:44px;display:flex;align-items:center;justify-content:center;border:1px solid #eceaf5;position:relative;">🔔<span style="position:absolute;top:9px;right:10px;width:8px;height:8px;background:#e8543f;border-radius:50%;"></span></div>', unsafe_allow_html=True)
    st.write("")

    if la:
        last_check_text = la["created_at"][:10]
        streak_text = f"{stats['count']} check{'s' if stats['count'] != 1 else ''} logged"
    else:
        last_check_text = "May 20"
        streak_text = "7 day streak"

    st.markdown(
        f"""
        <div class="hero">
            <h1>Good morning, {user['username']}</h1>
            <span class="pill">🗓️ Last check {last_check_text}</span>
            <span class="pill">🔥 {streak_text}</span>
            <div class="hero-icon">🧠</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("✨ Start new assessment", key="start_assessment_btn"):
        st.session_state.pending_nav = NAV_OPTIONS[1]
        st.rerun()

    if not la:
        st.info("👋 **This is sample data** — you haven't run a real assessment yet. Everything below is a preview to show the layout. Click **Start new assessment** above to get your actual result.")

    if model_error:
        st.markdown(
            f"""<div class="setup-warning">⚠️ <b>Model not loaded</b> — {'stroke_model.pkl' if model_error=='missing_model' else 'scaler.pkl'}
            is missing from the app folder.</div>""",
            unsafe_allow_html=True,
        )
        st.write("")

    # --- 4 vital cards (matches inspo exactly; illustrative demo values —
    # this dataset doesn't include wearable vitals) ---
    cols = st.columns(4)
    for col, v in zip(cols, DEMO_VITALS):
        arrow = "↗" if v["dir"] == "up" else "↘"
        with col:
            st.markdown(
                f"""<div class="vital-card {v['css']}">
                    <div class="vital-top"><span>{v['icon']}</span><span class="vital-delta">{arrow} {v['delta']}</span></div>
                    <div><div class="vital-value">{v['value']}</div><div class="vital-label">{v['label']}</div></div>
                    <div class="vital-bar"><span></span></div>
                </div>""",
                unsafe_allow_html=True,
            )

    # --- risk panel: real data once an assessment has been run, sample
    # data (matching inspo exactly) before that ---
    gauge_col, factors_col, rec_col = st.columns([1.1, 1.6, 1.3])

    if la:
        pct = la["risk_pct"]
        risk_label = "Elevated" if la["at_risk"] else "Low"
        factor_rows = [(s, "Flagged") for s in la["symptoms"]] or [("No symptoms flagged", "")]
        recs = [RECOMMENDATION_RULES[k] for k in la["symptom_keys"] if k in RECOMMENDATION_RULES]
        recs = (recs + DEFAULT_RECS)[:4] if recs else DEFAULT_RECS
    else:
        pct = SAMPLE_RISK["pct"]
        risk_label = SAMPLE_RISK["label"]
        factor_rows = SAMPLE_RISK["factors"]
        recs = SAMPLE_RISK["recs"]

    with gauge_col, st.container(key="risk_gauge"):
        st.markdown(
            f"""
            <svg width="150" height="150" viewBox="0 0 150 150" style="display:block;margin:0 auto;">
              <circle cx="75" cy="75" r="62" fill="none" stroke="#ffffff20" stroke-width="14"/>
              <circle cx="75" cy="75" r="62" fill="none" stroke="#f2a93b" stroke-width="14"
                      stroke-dasharray="{2*3.14159*62}"
                      stroke-dashoffset="{(1-pct/100)*2*3.14159*62}"
                      stroke-linecap="round" transform="rotate(-90 75 75)"/>
              <text x="75" y="70" text-anchor="middle" fill="#ffffff" font-size="28" font-weight="800" font-family="Inter">{pct}%</text>
              <text x="75" y="92" text-anchor="middle" fill="#b9b3d6" font-size="13" font-family="Inter">risk</text>
            </svg>
            <div class="risk-label">{risk_label}</div>
            """,
            unsafe_allow_html=True,
        )

    with factors_col, st.container(key="risk_factors"):
        st.markdown('<h3>Key factors</h3>', unsafe_allow_html=True)
        for name, level in factor_rows:
            level_html = f'<span class="lvl-high">{level}</span>' if level else ""
            st.markdown(f'<div class="factor-row"><span>{name}</span>{level_html}</div>', unsafe_allow_html=True)

    with rec_col, st.container(key="risk_rec"):
        st.markdown('<h3>Recommendations</h3>', unsafe_allow_html=True)
        for r in recs:
            st.markdown(f'<div class="rec-btn">{r}</div>', unsafe_allow_html=True)

    if not la:
        st.caption("Showing sample data above. Run a real check under **Risk check** to replace this with your result.")
    else:
        history = db.get_history(user["id"], limit=8)
        if len(history) > 1:
            st.markdown("#### Your past checks")
            for h in history:
                risk_word = "⚠️ At risk" if h["at_risk"] else "✅ Not at risk"
                st.markdown(
                    f"- {h['created_at'][:16].replace('T', ' ')} — age {h['age']}, {h['gender']} — "
                    f"{risk_word} ({h['risk_pct']}%) — {len(h['symptoms'])} symptom(s) flagged"
                )

# ===========================================================================
# RISK CHECK
# ===========================================================================
elif st.session_state.page == NAV_OPTIONS[1]:
    st.markdown('<div class="hero"><h1>Stroke Risk Assessment</h1><span class="pill">🩺 Age, gender & symptoms</span><div class="hero-icon">🩺</div></div>', unsafe_allow_html=True)

    if model_error:
        st.markdown(
            f"""<div class="setup-warning">⚠️ <b>Model not loaded</b> — {'stroke_model.pkl' if model_error=='missing_model' else 'scaler.pkl'}
            is missing. Add it next to <code>app.py</code> to enable predictions.</div>""",
            unsafe_allow_html=True,
        )
        st.write("")

    with st.form("stroke_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 1, 100, 30)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        st.caption("Gender is recorded in your history but isn't used by the model — the training data didn't include it.")

        st.markdown("**Symptoms**")
        cols = st.columns(3)
        checked_keys = []
        checked_labels = []
        symptom_values = []
        for i, (key, label) in enumerate(SYMPTOMS):
            with cols[i % 3]:
                checked = st.checkbox(label, key=f"sym_{key}")
            symptom_values.append(int(checked))
            if checked:
                checked_keys.append(key)
                checked_labels.append(label)

        submitted = st.form_submit_button("🔍 Predict Stroke Risk", use_container_width=True)

        if submitted:
            if model_error:
                st.error("Can't run a prediction — the model file is missing. See the notice above.")
            else:
                at_risk, risk_pct = run_prediction(age, symptom_values, model, scaler)
                db.save_assessment(
                    user["id"], age, gender, checked_labels, checked_keys, at_risk, risk_pct,
                )
                if at_risk:
                    st.error(f"⚠️ At risk of stroke — model confidence {risk_pct}%")
                else:
                    st.success(f"✅ Not at risk — model confidence {100-risk_pct}%")
                st.caption("See the Dashboard tab for the full breakdown.")

# ===========================================================================
# NEWS
# ===========================================================================
elif st.session_state.page == NAV_OPTIONS[2]:
    import requests

    st.markdown('<div class="hero"><h1>Latest Stroke News</h1><div class="hero-icon">📰</div></div>', unsafe_allow_html=True)

    news_api_key = st.secrets.get("NEWS_API_KEY", "") if hasattr(st, "secrets") else ""
    if not news_api_key:
        st.warning("Add `NEWS_API_KEY` to `.streamlit/secrets.toml` (or the Streamlit Cloud app secrets) to enable this tab.")
    else:
        if st.button("🔄 Refresh news"):
            try:
                resp = requests.get(f"https://newsapi.org/v2/everything?q=stroke&apiKey={news_api_key}")
                articles = resp.json().get("articles", [])[:5]
                for a in articles:
                    st.subheader(a["title"])
                    st.write(a.get("description", ""))
                    st.markdown(f"[Read more]({a['url']})")
                    st.markdown("---")
            except Exception:
                st.error("Failed to load news.")

# ===========================================================================
# CHATBOT
# ===========================================================================
elif st.session_state.page == NAV_OPTIONS[3]:
    st.markdown('<div class="hero"><h1>Chat with Stroke Assistant</h1><div class="hero-icon">💬</div></div>', unsafe_allow_html=True)

    hf_email = st.secrets.get("HF_EMAIL", "") if hasattr(st, "secrets") else ""
    hf_pass = st.secrets.get("HF_PASS", "") if hasattr(st, "secrets") else ""

    if not hf_email or not hf_pass:
        st.warning(
            "Chatbot credentials aren't configured. Add `HF_EMAIL` and `HF_PASS` to "
            "`.streamlit/secrets.toml` locally, or to your app's Secrets in Streamlit Cloud — "
            "**don't hardcode them in app.py**, especially since this is going in a public repo."
        )
    else:
        from hugchat import hugchat
        from hugchat.login import Login

        BASE_PROMPT = "Hello, I would like to ask about stroke prevention and symptoms. "

        @st.cache_resource
        def get_chatbot():
            sign = Login(hf_email, hf_pass)
            cookies = sign.login()
            return hugchat.ChatBot(cookies=cookies.get_dict())

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I help you regarding stroke?"}]
        if "chat_primed" not in st.session_state:
            st.session_state.chat_primed = False

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        if prompt := st.chat_input("Type your question about stroke here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        bot = get_chatbot()
                        full_prompt = prompt if st.session_state.chat_primed else BASE_PROMPT + prompt
                        st.session_state.chat_primed = True
                        response = str(bot.chat(full_prompt)).strip("`")
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error("Something went wrong with the chatbot.")
                        st.error(str(e))
