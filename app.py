# ============================================================
#  StreamMyMood — app.py  v2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time, base64
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

# ── LOAD LOGO AS BASE64 ───────────────────────────────────────
def get_logo_b64():
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = get_logo_b64()

def logo_html(width=220):
    if LOGO_B64:
        return f'<div style="text-align:center;margin-bottom:1.5rem"><img src="data:image/png;base64,{LOGO_B64}" width="{width}"/></div>'
    return '<div style="text-align:center;font-size:2rem;font-weight:800;color:#E91E63;margin-bottom:1.5rem">StreamMyMood</div>'

# ── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp, [data-testid="stAppViewContainer"] {
    background-color: #8B0000 !important;
    color: white !important;
  }
  [data-testid="stHeader"] { background-color: #8B0000 !important; }
  section[data-testid="stSidebar"] { display: none !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container {
    padding-top: 2rem !important;
    max-width: 1100px;
    margin: 0 auto;
  }
  h1,h2,h3,p,label,div,span {
    color: white !important;
    font-family: 'Segoe UI', Arial, sans-serif;
  }

  /* Buttons */
  .stButton { display: flex; justify-content: center; }
  .stButton > button {
    background: linear-gradient(135deg, #5c0000, #a00000) !important;
    color: white !important;
    border: 2px solid rgba(255,255,255,0.3) !important;
    border-radius: 30px !important;
    padding: 0.7rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    min-width: 220px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #a00000, #c00000) !important;
    border-color: white !important;
    transform: translateY(-2px) !important;
  }

  /* Radio buttons */
  div[data-testid="stRadio"] > label { display: none !important; }
  div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.5rem !important;
    align-items: center !important;
  }
  div[data-testid="stRadio"] label {
    background: rgba(0,0,0,0.25) !important;
    border: 2px solid rgba(255,255,255,0.25) !important;
    border-radius: 15px !important;
    padding: 0.8rem 1.5rem !important;
    width: 480px !important;
    max-width: 90vw !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-align: center !important;
    font-size: 1rem !important;
    display: block !important;
  }
  div[data-testid="stRadio"] label:hover {
    background: rgba(0,0,0,0.45) !important;
    border-color: white !important;
  }
  div[data-testid="stRadio"] label[data-checked="true"],
  div[data-testid="stRadio"] input:checked + label {
    background: rgba(0,0,0,0.5) !important;
    border-color: white !important;
    font-weight: 700 !important;
  }

  /* Progress bar */
  .prog-wrap {
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    height: 8px;
    margin: 0.5rem auto 2rem auto;
    max-width: 600px;
  }
  .prog-fill {
    background: white;
    height: 8px;
    border-radius: 10px;
    transition: width 0.4s ease;
  }

  /* Movie cards */
  .movie-card {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 1rem;
    height: 100%;
  }
  .movie-card img {
    width: 100%;
    border-radius: 10px;
    margin-bottom: 0.7rem;
    object-fit: cover;
    height: 300px;
  }
  .movie-title {
    font-size: 1rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.3rem !important;
  }
  .movie-meta {
    font-size: 0.8rem !important;
    color: rgba(255,255,255,0.7) !important;
    margin-bottom: 0.4rem !important;
  }
  .movie-overview {
    font-size: 0.82rem !important;
    color: rgba(255,255,255,0.85) !important;
    height: 110px;
    overflow-y: auto;
    line-height: 1.5;
  }
  .stars { color: #FFD700; font-size: 0.9rem; }

  /* Center text helpers */
  .center { text-align: center !important; }
  .big { font-size: 1.3rem !important; font-weight: 600 !important; }
  .sub { font-size: 1rem !important; color: rgba(255,255,255,0.8) !important; }

  /* Loading */
  .clapper-anim {
    font-size: 4rem;
    display: block;
    text-align: center;
    animation: clap 1s infinite alternate;
  }
  @keyframes clap {
    0%   { transform: rotate(-10deg) scale(1); }
    100% { transform: rotate(10deg)  scale(1.15); }
  }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
CONTENT_FILE  = "StreamMyMood_Content_Database.xlsx"
TRAINING_FILE = "StreamMyMood_Training_Database.xlsx"
MODEL_FILE    = "streammymood_model.pkl"

# Runtime upper bounds (minutes) — content duration, not binge total
RUNTIME_LIMITS = {
    "עד 30 דקות":   30,
    "30–60 דקות":   60,
    "60–120 דקות":  120,
    "מעל שעתיים":   9999,
}

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

# ── DATA ──────────────────────────────────────────────────────
@st.cache_data
def load_content():
    df = pd.read_excel(CONTENT_FILE)

    # Parse runtime — extract integer minutes from "95 min" etc.
    def parse_rt(v):
        try:
            return int(str(v).replace("min","").replace("דקות","").strip())
        except:
            return None

    df["Runtime_Int"] = df["Runtime"].apply(parse_rt)

    # For series: Runtime_Int = episode length (already per-episode in DB)
    # This is correct — user says "I have 60 min" = single episode or movie

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    mean_rating  = df["Rating"].mean()
    df["Rating"].fillna(mean_rating, inplace=True)
    df["Major_Awards_Won"] = pd.to_numeric(df["Major_Awards_Won"], errors="coerce").fillna(0)

    # Build title→ID map
    df["ID"] = df["ID"].astype(str)
    return df

# ── MODEL ─────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model(content_df):
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    train_df = pd.read_excel(TRAINING_FILE)

    # Rename columns to match feature names
    col_map = {
        "mood":              "User_Mood",
        "time_available":    "Duration_Limit",
        "watching_with":     "Watch_Group",
        "awards_preference": "Award_Importance",
        "rating_influence":  "Review_Importance",
        "chosen_content":    "Content_Title",
    }
    train_df.rename(columns={k:v for k,v in col_map.items() if k in train_df.columns}, inplace=True)

    # Keep only titles that exist in content DB
    valid = set(content_df["Title"].tolist())
    train_df = train_df[train_df["Content_Title"].isin(valid)].copy()

    if len(train_df) < 10:
        st.error("אין מספיק נתוני אימון.")
        st.stop()

    features = ["Duration_Limit","Watch_Group","User_Mood","Review_Importance","Award_Importance"]

    encoders = {}
    X = pd.DataFrame()
    for f in features:
        le = LabelEncoder()
        X[f] = le.fit_transform(train_df[f].astype(str))
        encoders[f] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df["Content_Title"].astype(str))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_val, y_val)

    data = {"model":clf,"encoders":encoders,"le_target":le_target,
            "features":features,"accuracy":acc}
    joblib.dump(data, MODEL_FILE)
    return data

# ── RECOMMENDATION ENGINE ─────────────────────────────────────
def get_recommendations(answers, content_df, model_data, seen_ids=None):
    seen_ids = seen_ids or set()
    clf, encoders = model_data["model"], model_data["encoders"]
    le_target, features = model_data["le_target"], model_data["features"]

    answer_map = {
        "Duration_Limit":    answers.get("time_available",""),
        "Watch_Group":       answers.get("watching_with",""),
        "User_Mood":         answers.get("mood",""),
        "Review_Importance": answers.get("review_importance",""),
        "Award_Importance":  answers.get("awards_preference",""),
    }

    row = {}
    for f in features:
        le  = encoders[f]
        val = answer_map[f]
        row[f] = le.transform([val])[0] if val in le.classes_ else 0

    X_in = pd.DataFrame([row])[features]
    proba = clf.predict_proba(X_in)[0]
    titles = le_target.inverse_transform(clf.classes_)
    model_scores = dict(zip(titles, proba))

    # Runtime limit — single content duration (episode or movie)
    max_rt      = RUNTIME_LIMITS.get(answers.get("time_available","מעל שעתיים"), 9999)
    wants_awards = "כן" in answers.get("awards_preference","")
    max_rating   = content_df["Rating"].max() or 10.0

    results = []
    for _, c in content_df.iterrows():
        cid = str(c["ID"])
        if cid in seen_ids:
            continue

        rt = c.get("Runtime_Int") or 9999

        # Hard runtime filter — episode/movie must fit in available time
        if rt > max_rt:
            continue

        ms             = model_scores.get(c["Title"], 0.0)
        norm_rating    = c["Rating"] / max_rating
        awards_score   = min(1.0, (c.get("Major_Awards_Won",0) or 0) / 10.0)

        final = 0.6*ms + 0.3*norm_rating + 0.1*awards_score

        # Boost award-winners if user wants awards
        if wants_awards and (c.get("Major_Awards_Won",0) or 0) > 0:
            final *= 1.3

        results.append({"row": c, "score": final, "id": cid})

    results.sort(key=lambda x: x["score"], reverse=True)

    # If < 4 results after hard filter → supplement from all content (no runtime filter)
    if len(results) < 4:
        extras = []
        for _, c in content_df.iterrows():
            cid = str(c["ID"])
            if cid in seen_ids or any(r["id"]==cid for r in results):
                continue
            ms          = model_scores.get(c["Title"], 0.0)
            norm_rating = c["Rating"] / max_rating
            awards_score= min(1.0,(c.get("Major_Awards_Won",0) or 0)/10.0)
            final       = 0.6*ms + 0.3*norm_rating + 0.1*awards_score
            extras.append({"row":c,"score":final,"id":cid})
        extras.sort(key=lambda x: x["score"], reverse=True)
        results += extras
        if results:
            st.info("לא מצאנו תוכן מדויק, אבל הנה התכנים הכי קרובים לבקשה שלך.")

    return results

# ── HELPERS ───────────────────────────────────────────────────
def stars(rating):
    filled = max(0, min(5, round(float(rating) / 2)))
    return "★" * filled + "☆" * (5 - filled)

def placeholder_img():
    """Return logo as fallback image src."""
    if LOGO_B64:
        return f"data:image/png;base64,{LOGO_B64}"
    return "https://via.placeholder.com/300x300/8B0000/ffffff?text=StreamMyMood"

def movie_card_html(c):
    poster  = c.get("Poster_URL","") or ""
    if not poster or str(poster).strip() in ["","nan","None"]:
        poster = placeholder_img()
    title   = c.get("Title","") or ""
    year    = int(c["Year"]) if c.get("Year") and str(c["Year"]) != "nan" else "—"
    runtime = c.get("Runtime","—") or "—"
    rating  = float(c.get("Rating",7.9) or 7.9)
    overview= c.get("Overview","") or ""

    return f"""
<div class="movie-card">
  <img src="{poster}" alt="{title}"
       onerror="this.onerror=null;this.src='{placeholder_img()}'"/>
  <div class="movie-title">{title}</div>
  <div class="movie-meta">{year} &nbsp;·&nbsp; {runtime}</div>
  <div class="stars">{stars(rating)} &nbsp;<span style="color:rgba(255,255,255,0.7);font-size:0.8rem">{rating:.1f}</span></div>
  <div class="movie-overview" style="margin-top:0.5rem">{overview}</div>
</div>"""

# ── SCREENS ───────────────────────────────────────────────────
def screen_welcome():
    st.markdown(logo_html(240), unsafe_allow_html=True)

    h = datetime.now().hour
    greeting = "בוקר טוב" if h < 12 else "צהריים טובים" if h < 17 else "ערב טוב" if h < 21 else "לילה טוב"

    st.markdown(f"""
    <div class="center" style="margin-bottom:0.5rem">
      <div class="big">{greeting}!</div>
    </div>
    <div class="center" style="margin-bottom:2.5rem">
      <div class="sub">ברוכים הבאים ל‑Stream My Mood<br>בואו נמצא לכם מה לראות.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("לחצו כאן כדי להתחיל!"):
            st.session_state.screen   = "quiz"
            st.session_state.q_index  = 0
            st.session_state.answers  = {}
            st.session_state.seen_ids = set()
            st.session_state.page     = 0
            st.session_state.ranked   = None
            st.rerun()


def screen_quiz():
    st.markdown(logo_html(160), unsafe_allow_html=True)

    q_idx = st.session_state.q_index
    q     = QUESTIONS[q_idx]
    total = len(QUESTIONS)
    pct   = int((q_idx / total) * 100)

    st.markdown(f"""
    <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='center' style='font-size:0.85rem;margin-bottom:1rem;color:rgba(255,255,255,0.6)'>שאלה {q_idx+1} מתוך {total}</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='center big' style='margin-bottom:1.5rem'>{q['text']}</div>",
                unsafe_allow_html=True)

    current = st.session_state.answers.get(q["id"])
    idx     = q["options"].index(current) if current in q["options"] else 0

    selected = st.radio("", q["options"], index=idx, key=f"r_{q_idx}")
    st.session_state.answers[q["id"]] = selected

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        label = "לשאלה הבאה" if q_idx < total-1 else "קבלו המלצות!"
        if st.button(label):
            if q_idx < total-1:
                st.session_state.q_index += 1
                st.rerun()
            else:
                st.session_state.screen = "loading"
                st.rerun()


def screen_loading():
    st.markdown(logo_html(160), unsafe_allow_html=True)

    placeholder = st.empty()
    for i in range(4):
        txt = LOADING_TEXTS[i % len(LOADING_TEXTS)]
        placeholder.markdown(f"""
        <div style="text-align:center;padding:3rem 0">
          <span class="clapper-anim">🎬</span>
          <div style="font-size:1.2rem;margin-top:1.5rem;color:rgba(255,255,255,0.9)">{txt}</div>
        </div>""", unsafe_allow_html=True)
        time.sleep(0.6)

    st.session_state.screen = "results"
    st.rerun()


def screen_results(content_df, model_data):
    st.markdown(logo_html(160), unsafe_allow_html=True)

    answers  = st.session_state.answers
    seen_ids = st.session_state.get("seen_ids", set())
    page     = st.session_state.get("page", 0)

    # Run model once and cache
    if not st.session_state.get("ranked"):
        st.session_state.ranked = get_recommendations(answers, content_df, model_data, set())

    ranked = st.session_state.ranked
    # Filter out already seen
    remaining = [r for r in ranked if r["id"] not in seen_ids]
    batch     = remaining[:4]

    if not batch:
        st.markdown("<div class='center big'>לא קיימות המלצות נוספות בהתאם להגדרות שלך.</div>",
                    unsafe_allow_html=True)
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            if st.button("התחל מחדש"):
                for k in ["screen","q_index","answers","seen_ids","page","ranked"]:
                    st.session_state.pop(k, None)
                st.rerun()
        return

    st.markdown("<div class='center' style='font-size:1.3rem;font-weight:700;margin-bottom:1.5rem'>ההמלצות שלנו עבורך</div>",
                unsafe_allow_html=True)

    cols = st.columns(4)
    for col, r in zip(cols, batch):
        with col:
            st.markdown(movie_card_html(r["row"]), unsafe_allow_html=True)
        seen_ids.add(r["id"])

    st.session_state.seen_ids = seen_ids
    st.markdown("<br><br>", unsafe_allow_html=True)

    next_avail = [r for r in ranked if r["id"] not in seen_ids]

    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if next_avail:
            if st.button("לא אהבתי את ההמלצות האלה, אשמח להמלצות נוספות"):
                st.rerun()
        else:
            st.markdown("<div class='center sub'>לא קיימות המלצות נוספות בהתאם להגדרות שלך.</div>",
                        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("התחל מחדש"):
            for k in ["screen","q_index","answers","seen_ids","page","ranked"]:
                st.session_state.pop(k, None)
            st.rerun()


# ── MAIN ──────────────────────────────────────────────────────
def main():
    if "screen" not in st.session_state:
        st.session_state.screen = "welcome"

    content_df = load_content()
    model_data = load_or_train_model(content_df)

    s = st.session_state.screen
    if   s == "welcome": screen_welcome()
    elif s == "quiz":    screen_quiz()
    elif s == "loading": screen_loading()
    elif s == "results": screen_results(content_df, model_data)

if __name__ == "__main__":
    main()
