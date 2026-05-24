# ============================================================
#  StreamMyMood — app.py
#  Streamlit + Random Forest recommendation system
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="StreamMyMood",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark Burgundy background */
  .stApp, [data-testid="stAppViewContainer"] {
    background-color: #2D0A16 !important;
    color: white !important;
  }
  [data-testid="stHeader"] { background-color: #2D0A16 !important; }
  section[data-testid="stSidebar"] { background-color: #1a0610 !important; }

  /* Hide default streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem !important; max-width: 1100px; }

  /* Typography */
  h1, h2, h3, p, label, div {
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
  }

  /* Primary buttons */
  .stButton > button {
    background: linear-gradient(135deg, #C2185B, #E91E63) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #E91E63, #F06292) !important;
    transform: translateY(-2px) !important;
  }

  /* Answer buttons */
  .answer-btn {
    background: rgba(255,255,255,0.08);
    border: 2px solid rgba(255,255,255,0.2);
    border-radius: 15px;
    padding: 1rem 1.5rem;
    margin: 0.4rem 0;
    width: 100%;
    color: white;
    font-size: 1rem;
    cursor: pointer;
    text-align: right;
    transition: all 0.2s;
  }
  .answer-btn:hover, .answer-btn.selected {
    background: linear-gradient(135deg, #C2185B, #E91E63);
    border-color: #E91E63;
    transform: translateX(-4px);
  }

  /* Progress bar */
  .progress-container {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    height: 8px;
    margin: 1rem 0 2rem 0;
  }
  .progress-fill {
    background: linear-gradient(90deg, #C2185B, #E91E63);
    height: 8px;
    border-radius: 10px;
    transition: width 0.4s ease;
  }

  /* Movie cards */
  .movie-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.2rem;
    height: 100%;
    transition: transform 0.2s;
  }
  .movie-card:hover { transform: translateY(-4px); }
  .movie-card img { width: 100%; border-radius: 10px; margin-bottom: 0.8rem; }
  .movie-title { font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem; }
  .movie-meta { font-size: 0.8rem; color: #ccc !important; margin-bottom: 0.5rem; }
  .movie-overview {
    font-size: 0.82rem;
    color: #ddd !important;
    height: 120px;
    overflow-y: auto;
    line-height: 1.5;
    padding-right: 4px;
  }
  .stars { color: #FFD700; font-size: 0.9rem; }

  /* Logo */
  .logo-text {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E91E63, #F48FB1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
  }
  .logo-sub {
    text-align: center;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5) !important;
    margin-bottom: 2rem;
  }

  /* Welcome screen */
  .welcome-box {
    text-align: center;
    padding: 3rem 2rem;
  }
  .greeting { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.5rem; }
  .subtitle { color: rgba(255,255,255,0.7) !important; margin-bottom: 2.5rem; font-size: 1rem; }

  /* Loading */
  .loading-box { text-align: center; padding: 4rem 2rem; }
  .clapper { font-size: 5rem; animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.15)} }
  .loading-text { font-size: 1.2rem; color: rgba(255,255,255,0.8) !important; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ─────────────────────────────────────────────────
CONTENT_FILE  = "StreamMyMood_Content_Database.xlsx"
TRAINING_FILE = "StreamMyMood_Training_Database.xlsx"
MODEL_FILE    = "streammymood_model.pkl"

QUESTIONS = [
    {
        "id": "time_available",
        "text": "כמה זמן פנוי יש לך לצפייה?",
        "options": ["עד 30 דקות", "30–60 דקות", "60–120 דקות", "מעל שעתיים"]
    },
    {
        "id": "watching_with",
        "text": "עם מי את/ה נמצא?",
        "options": ["לבד", "בן / בת זוג", "משפחה", "חברים", "ילדים"]
    },
    {
        "id": "mood",
        "text": "מה המצב רוח שלך כרגע?",
        "options": [
            "קליל וחיפשתי משהו שיעשה לי טוב על הלב",
            "משועמם וחיפשתי משהו שיעורר אותי",
            "סקרן וחיפשתי סיפור להישאב אליו",
            "רגיש וחיפשתי משהו שייגע בי",
            "עמוס וחיפשתי פשוט 'להיעלם' בתוך עולם אחר",
            "מרוקן ורוצה להעביר את הזמן בלי לחשוב",
        ]
    },
    {
        "id": "review_importance",
        "text": "עד כמה חשובות לך ביקורות?",
        "options": ["חשובות מאוד!", "לא באמת משנה לי"]
    },
    {
        "id": "awards_preference",
        "text": "האם חשוב שהתוכן זכה בפרסים?",
        "options": ["כן, רק זוכי פרסים", "לא קריטי עבורי"]
    },
]

LOADING_TEXTS = [
    "מכינים המלצה שתתאים לך בול…",
    "בודקים את מצב הרוח שלך…",
    "סורקים את התכנים…",
    "מחשבים התאמה אישית במיוחד בשבילך…",
]

RUNTIME_LIMITS = {
    "עד 30 דקות": 30,
    "30–60 דקות": 60,
    "60–120 דקות": 120,
    "מעל שעתיים": 9999,
}


# ── DATA & MODEL ──────────────────────────────────────────────
@st.cache_data
def load_content():
    df = pd.read_excel(CONTENT_FILE)
    # Parse runtime to int
    def parse_rt(v):
        try:
            return int(str(v).replace(" min","").replace("min","").strip())
        except:
            return None
    df["Runtime_Int"] = df["Runtime"].apply(parse_rt)
    # Fill missing rating with mean
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Rating"].fillna(df["Rating"].mean(), inplace=True)
    # Awards
    df["Major_Awards_Won"] = pd.to_numeric(df["Major_Awards_Won"], errors="coerce").fillna(0)
    return df


@st.cache_resource
def load_or_train_model(content_df):
    """Load saved model or train fresh one."""

    if os.path.exists(MODEL_FILE):
        model_data = joblib.load(MODEL_FILE)
        return model_data

    # ── Load training data ──────────────────────────────────
    train_df = pd.read_excel(TRAINING_FILE)

    # Map column names from training table
    col_map = {
        "mood":              "User_Mood",
        "time_available":    "Duration_Limit",
        "watching_with":     "Watch_Group",
        "awards_preference": "Award_Importance",
        "rating_influence":  "Review_Importance",
        "chosen_content":    "Content_Title",
    }
    train_df.rename(columns=col_map, inplace=True)

    # Keep only rows where Content_Title exists in content DB
    valid_titles = set(content_df["Title"].tolist())
    train_df = train_df[train_df["Content_Title"].isin(valid_titles)].copy()

    if len(train_df) < 10:
        st.error("אין מספיק נתוני אימון — ודאי שהקובץ Training_Database.xlsx תקין.")
        st.stop()

    # ── Encode features ─────────────────────────────────────
    features = ["Duration_Limit", "Watch_Group", "User_Mood",
                "Review_Importance", "Award_Importance"]

    encoders = {}
    X = pd.DataFrame()
    for f in features:
        le = LabelEncoder()
        X[f] = le.fit_transform(train_df[f].astype(str))
        encoders[f] = le

    # Target: content title → integer label
    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df["Content_Title"].astype(str))

    # ── Train ────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_val, y_val)

    model_data = {
        "model":      clf,
        "encoders":   encoders,
        "le_target":  le_target,
        "features":   features,
        "accuracy":   accuracy,
    }
    joblib.dump(model_data, MODEL_FILE)
    return model_data


def get_recommendations(answers, content_df, model_data, seen_ids=None):
    """Run the full scoring pipeline and return ranked content rows."""
    seen_ids = seen_ids or set()
    clf       = model_data["model"]
    encoders  = model_data["encoders"]
    le_target = model_data["le_target"]
    features  = model_data["features"]

    # Build input vector — handle unseen labels gracefully
    row = {}
    answer_map = {
        "Duration_Limit":   answers.get("time_available", ""),
        "Watch_Group":      answers.get("watching_with", ""),
        "User_Mood":        answers.get("mood", ""),
        "Review_Importance":answers.get("review_importance", ""),
        "Award_Importance": answers.get("awards_preference", ""),
    }
    for f in features:
        le = encoders[f]
        val = answer_map[f]
        if val in le.classes_:
            row[f] = le.transform([val])[0]
        else:
            row[f] = 0  # fallback

    X_input = pd.DataFrame([row])[features]

    # Probabilities for all known classes
    proba = clf.predict_proba(X_input)[0]
    class_labels = le_target.inverse_transform(clf.classes_)

    # Build scores dict: title → model_score
    model_scores = {title: p for title, p in zip(class_labels, proba)}

    # Scoring per content row
    max_rt = RUNTIME_LIMITS.get(answers.get("time_available","מעל שעתיים"), 9999)
    wants_awards = "כן" in answers.get("awards_preference", "")
    max_rating = content_df["Rating"].max() or 10.0

    results = []
    for _, c in content_df.iterrows():
        if c["ID"] in seen_ids:
            continue

        # Runtime filter (soft — applied after scoring)
        rt = c.get("Runtime_Int") or 999
        passes_rt = rt <= max_rt

        ms = model_scores.get(c["Title"], 0.0)
        normalized_rating = c["Rating"] / max_rating
        awards_won = c.get("Major_Awards_Won", 0) or 0
        awards_score = min(1.0, awards_won / 10.0)

        final = 0.6 * ms + 0.3 * normalized_rating + 0.1 * awards_score
        results.append({
            "row":        c,
            "score":      final,
            "passes_rt":  passes_rt,
            "awards_won": awards_won,
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Apply award filter preference (boost, not hard filter)
    if wants_awards:
        with_awards    = [r for r in results if r["awards_won"] > 0 and r["passes_rt"]]
        without_awards = [r for r in results if r["awards_won"] == 0 and r["passes_rt"]]
        filtered = with_awards + without_awards
    else:
        filtered = [r for r in results if r["passes_rt"]]

    # If < 4 after runtime filter, supplement from unfiltered
    if len(filtered) < 4:
        extras = [r for r in results if r not in filtered]
        filtered += extras

    return [r["row"] for r in filtered]


# ── UI HELPERS ────────────────────────────────────────────────
def logo():
    st.markdown('<div class="logo-text">🎬 StreamMyMood</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">גלו מה לראות הערב</div>', unsafe_allow_html=True)

def greeting():
    h = datetime.now().hour
    if h < 12:   return "בוקר טוב! ☀️"
    elif h < 17: return "צהריים טובים! 🌤️"
    elif h < 21: return "ערב טוב! 🌙"
    else:         return "לילה טוב! ⭐"

def stars(rating):
    filled = round(rating / 2)
    return "⭐" * filled + "☆" * (5 - filled)

def movie_card(c):
    poster = c.get("Poster_URL","")
    if not poster or str(poster) == "nan":
        poster = "https://via.placeholder.com/300x450/2D0A16/E91E63?text=StreamMyMood"
    title    = c.get("Title","")
    year     = int(c.get("Year",0)) if c.get("Year") else "—"
    runtime  = c.get("Runtime","—")
    rating   = c.get("Rating",7.9) or 7.9
    overview = c.get("Overview","") or ""

    st.markdown(f"""
    <div class="movie-card">
      <img src="{poster}" alt="{title}" onerror="this.src='https://via.placeholder.com/300x450/2D0A16/E91E63?text=No+Image'"/>
      <div class="movie-title">{title}</div>
      <div class="movie-meta">{year} &nbsp;·&nbsp; {runtime}</div>
      <div class="stars">{stars(rating)} &nbsp;<span style="color:#ccc;font-size:0.8rem">{rating:.1f}</span></div>
      <div class="movie-overview" style="margin-top:0.6rem">{overview}</div>
    </div>
    """, unsafe_allow_html=True)


# ── SCREEN FUNCTIONS ──────────────────────────────────────────
def screen_welcome():
    logo()
    st.markdown(f"""
    <div class="welcome-box">
      <div class="greeting">{greeting()}</div>
      <div class="subtitle">ברוכים הבאים ל‑Stream My Mood<br>בואו נמצא לכם מה לראות.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("לחצו כאן כדי להתחיל! 🎬"):
            st.session_state.screen   = "quiz"
            st.session_state.q_index  = 0
            st.session_state.answers  = {}
            st.session_state.seen_ids = set()
            st.session_state.page     = 0
            st.rerun()


def screen_quiz():
    logo()
    q_idx = st.session_state.q_index
    q     = QUESTIONS[q_idx]
    total = len(QUESTIONS)

    # Progress bar
    pct = int((q_idx / total) * 100)
    st.markdown(f"""
    <div class="progress-container">
      <div class="progress-fill" style="width:{pct}%"></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"שאלה {q_idx+1} מתוך {total}")

    st.markdown(f"### {q['text']}")
    st.markdown("<br>", unsafe_allow_html=True)

    current_answer = st.session_state.answers.get(q["id"])

    # Answer buttons via Streamlit radio (styled)
    selected = st.radio(
        label="",
        options=q["options"],
        index=q["options"].index(current_answer) if current_answer in q["options"] else 0,
        key=f"radio_{q_idx}",
        label_visibility="collapsed"
    )
    st.session_state.answers[q["id"]] = selected

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        btn_label = "לשאלה הבאה ←" if q_idx < total - 1 else "קבלו המלצות! 🎬"
        if st.button(btn_label):
            if q_idx < total - 1:
                st.session_state.q_index += 1
                st.rerun()
            else:
                st.session_state.screen = "loading"
                st.rerun()


def screen_loading():
    logo()
    txt = LOADING_TEXTS[int(time.time()) % len(LOADING_TEXTS)]
    st.markdown(f"""
    <div class="loading-box">
      <div class="clapper">🎬</div>
      <div class="loading-text">{txt}</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    st.session_state.screen = "results"
    st.rerun()


def screen_results(content_df, model_data):
    logo()
    answers  = st.session_state.answers
    seen_ids = st.session_state.get("seen_ids", set())
    page     = st.session_state.get("page", 0)

    # Get all ranked content
    ranked = get_recommendations(answers, content_df, model_data, seen_ids)

    # Slice 4 per page
    start = page * 4
    batch = ranked[start:start + 4]

    if not batch:
        st.markdown("### 😔 לא קיימות המלצות נוספות בהתאם להגדרות שלך.")
        if st.button("התחל מחדש"):
            for k in ["screen","q_index","answers","seen_ids","page"]:
                st.session_state.pop(k, None)
            st.rerun()
        return

    st.markdown("## 🎯 ההמלצות שלנו עבורך")
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)
    for col, c in zip(cols, batch):
        with col:
            movie_card(c)
        seen_ids.add(c["ID"])

    st.session_state.seen_ids = seen_ids
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # More recommendations button
        next_batch = ranked[start + 4:start + 8]
        if next_batch:
            if st.button("אשמח להמלצות נוספות 🔄"):
                st.session_state.page += 1
                st.rerun()
        else:
            st.markdown(
                "<p style='text-align:center;color:rgba(255,255,255,0.5)'>"
                "לא קיימות המלצות נוספות בהתאם להגדרות שלך.</p>",
                unsafe_allow_html=True
            )

        if st.button("התחל מחדש 🔁"):
            for k in ["screen","q_index","answers","seen_ids","page"]:
                st.session_state.pop(k, None)
            st.rerun()


# ── MAIN ──────────────────────────────────────────────────────
def main():
    # Init session state
    if "screen" not in st.session_state:
        st.session_state.screen = "welcome"

    # Load data & model
    content_df = load_content()
    model_data = load_or_train_model(content_df)

    screen = st.session_state.screen

    if screen == "welcome":
        screen_welcome()
    elif screen == "quiz":
        screen_quiz()
    elif screen == "loading":
        screen_loading()
    elif screen == "results":
        screen_results(content_df, model_data)


if __name__ == "__main__":
    main()
