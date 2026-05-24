# ============================================================
#  StreamMyMood — app.py  v6
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time, base64
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="StreamMyMood",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── ASSETS ────────────────────────────────────────────────────
def file_b64(path):
    if os.path.exists(path):
        with open(path,"rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = file_b64("logo - StreamMyMood.png")
BG_B64   = file_b64("background.png")

def logo_html(width=340):
    if LOGO_B64:
        return (f'<div style="text-align:center;margin-bottom:1.4rem">'
                f'<img src="data:image/png;base64,{LOGO_B64}" '
                f'width="{width}" style="max-width:90vw"/></div>')
    return '<div style="text-align:center;font-size:2.5rem;font-weight:900;color:#ff4d6d;margin-bottom:1.4rem">StreamMyMood</div>'

def placeholder_src():
    if LOGO_B64:
        return f"data:image/png;base64,{LOGO_B64}"
    return "https://via.placeholder.com/300x420/3d0018/ffffff?text=StreamMyMood"

# ── CSS ───────────────────────────────────────────────────────
bg_css = ""
if BG_B64:
    bg_css = f"""
  background-image: url("data:image/png;base64,{BG_B64}") !important;
  background-size: cover !important;
  background-position: center top !important;
  background-attachment: fixed !important;
"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700;800&display=swap');

html, body, * {{
  font-family: 'Assistant', Arial, sans-serif !important;
  direction: rtl !important;
  box-sizing: border-box;
}}

/* Background */
.stApp, [data-testid="stAppViewContainer"] {{
  {bg_css}
  background-color: #2d0012 !important;
  min-height: 100vh;
  color: white !important;
}}

/* Dark overlay for readability */
[data-testid="stAppViewContainer"]::before {{
  content: '';
  position: fixed; inset: 0;
  background: rgba(15,0,6,0.75);
  pointer-events: none;
  z-index: 0;
}}

[data-testid="stHeader"] {{ background: transparent !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}

.block-container {{
  padding-top: 2rem !important;
  max-width: 1150px !important;
  margin: 0 auto !important;
  position: relative;
  z-index: 1;
}}

h1,h2,h3,p,label,div,span {{ color: white !important; }}

/* ── FORCE ALL CONTENT CENTER ── */
[data-testid="stVerticalBlock"],
[data-testid="column"] {{
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
}}

/* ── Buttons ── */
.stButton {{
  display: flex !important;
  justify-content: center !important;
  width: 100% !important;
}}
.stButton > button {{
  background: linear-gradient(135deg, #7a0025, #b30030) !important;
  color: white !important;
  border: 1.5px solid rgba(255,255,255,0.25) !important;
  border-radius: 30px !important;
  padding: 0.75rem 2.8rem !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  font-family: 'Assistant', sans-serif !important;
  min-width: 260px !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 18px rgba(180,0,50,0.4) !important;
  direction: rtl !important;
}}
.stButton > button:hover {{
  background: linear-gradient(135deg, #b30030, #d4003a) !important;
  transform: translateY(-2px) !important;
}}

/* ── Hide radio button circles + top empty option ── */
div[data-testid="stRadio"] input[type="radio"] {{ display: none !important; }}
div[data-testid="stRadio"] > label {{ display: none !important; }}
div[data-testid="stRadio"] > div {{
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  gap: 0.55rem !important;
  width: 100% !important;
}}
div[data-testid="stRadio"] > div > label:first-child {{
  display: none !important;
}}
div[data-testid="stRadio"] label {{
  background: rgba(255,255,255,0.07) !important;
  border: 1.5px solid rgba(255,255,255,0.2) !important;
  border-radius: 14px !important;
  padding: 0.85rem 2rem !important;
  width: 520px !important;
  max-width: 88vw !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  text-align: center !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  direction: rtl !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}}
div[data-testid="stRadio"] label:hover {{
  background: rgba(180,0,50,0.28) !important;
  border-color: rgba(255,100,130,0.6) !important;
}}
div[data-testid="stRadio"] label[data-checked="true"] {{
  background: rgba(180,0,50,0.5) !important;
  border-color: rgba(255,140,160,0.9) !important;
  font-weight: 800 !important;
}}

/* ── Progress bar ── */
.prog-outer {{
  display: flex;
  justify-content: center;
  width: 100%;
  margin: 0.3rem 0 1.8rem 0;
}}
.prog-wrap {{
  background: rgba(255,255,255,0.12);
  border-radius: 10px; height: 6px;
  width: 560px; max-width: 88vw;
}}
.prog-fill {{
  background: linear-gradient(90deg, #b30030, #ff4d6d);
  height: 6px; border-radius: 10px;
  transition: width 0.4s ease;
  box-shadow: 0 0 8px rgba(255,77,109,0.5);
}}

/* ── Movie cards ── */
.movie-card {{
  background: rgba(10,0,5,0.65);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 1rem;
  backdrop-filter: blur(8px);
  transition: transform 0.2s, box-shadow 0.2s;
  text-align: right;
  direction: rtl;
  height: 100%;
}}
.movie-card:hover {{
  transform: translateY(-4px);
  box-shadow: 0 8px 28px rgba(180,0,50,0.35);
}}
.movie-poster {{
  width: 100%; border-radius: 10px;
  margin-bottom: 0.7rem;
  object-fit: cover; height: 280px;
  background: #1a0008;
  display: block;
}}
.movie-title {{
  font-size: 1rem !important; font-weight: 800 !important;
  margin-bottom: 0.2rem !important; line-height: 1.3 !important;
}}
.movie-meta {{
  font-size: 0.8rem !important;
  color: rgba(255,255,255,0.6) !important;
  margin-bottom: 0.25rem !important;
}}
.movie-platform {{
  font-size: 0.8rem !important; color: #ff9ab0 !important;
  font-weight: 700 !important; margin-bottom: 0.3rem !important;
}}
.movie-stars {{ color: #FFD700; font-size: 0.88rem; margin-bottom: 0.4rem; }}
.movie-overview {{
  font-size: 0.82rem !important;
  color: rgba(255,255,255,0.8) !important;
  height: 100px; overflow-y: auto; line-height: 1.55;
}}

/* ── Film reel animation ── */
.reel-wrap {{ text-align: center; padding: 2.5rem 0; }}
.reel-svg {{
  display: inline-block;
  animation: spin 1.8s linear infinite;
  filter: drop-shadow(0 0 12px rgba(200,0,50,0.6));
}}
@keyframes spin {{
  from {{ transform: rotate(0deg); }}
  to   {{ transform: rotate(360deg); }}
}}
.loading-txt {{
  font-size: 1.15rem !important;
  color: rgba(255,255,255,0.88) !important;
  margin-top: 1.2rem !important;
  font-weight: 600 !important;
}}

.center {{ text-align: center !important; }}
.big    {{ font-size: 1.35rem !important; font-weight: 800 !important; }}
.sub    {{ font-size: 1rem !important; color: rgba(255,255,255,0.78) !important; line-height: 1.7; }}
</style>
""", unsafe_allow_html=True)

# ── FILM REEL SVG ─────────────────────────────────────────────
REEL_SVG = """
<svg class="reel-svg" width="100" height="100" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
  <!-- Outer ring -->
  <circle cx="50" cy="50" r="46" stroke="#cc0033" stroke-width="5" fill="#1a0008"/>
  <!-- Middle ring -->
  <circle cx="50" cy="50" r="30" stroke="#cc0033" stroke-width="3" fill="#0d0005"/>
  <!-- Center hub -->
  <circle cx="50" cy="50" r="10" fill="#cc0033"/>
  <!-- Spokes -->
  <line x1="50" y1="40" x2="50" y2="22" stroke="#cc0033" stroke-width="3" stroke-linecap="round"/>
  <line x1="50" y1="60" x2="50" y2="78" stroke="#cc0033" stroke-width="3" stroke-linecap="round"/>
  <line x1="40" y1="50" x2="22" y2="50" stroke="#cc0033" stroke-width="3" stroke-linecap="round"/>
  <line x1="60" y1="50" x2="78" y2="50" stroke="#cc0033" stroke-width="3" stroke-linecap="round"/>
  <line x1="43" y1="43" x2="30" y2="30" stroke="#cc0033" stroke-width="3" stroke-linecap="round"/>
  <line x1="57" y1="57" x2="70" y2="70" stroke="#cc0033" stroke-width="3" stroke-linecap="round"/>
  <line x1="57" y1="43" x2="70" y2="30" stroke="#cc0033" stroke-width="3" stroke-linecap="round"/>
  <line x1="43" y1="57" x2="30" y2="70" stroke="#cc0033" stroke-width="3" stroke-linecap="round"/>
  <!-- Sprocket holes -->
  <circle cx="50" cy="8"  r="5" fill="#cc0033"/>
  <circle cx="50" cy="92" r="5" fill="#cc0033"/>
  <circle cx="8"  cy="50" r="5" fill="#cc0033"/>
  <circle cx="92" cy="50" r="5" fill="#cc0033"/>
  <circle cx="22" cy="22" r="5" fill="#cc0033"/>
  <circle cx="78" cy="78" r="5" fill="#cc0033"/>
  <circle cx="78" cy="22" r="5" fill="#cc0033"/>
  <circle cx="22" cy="78" r="5" fill="#cc0033"/>
</svg>"""

# ── CONSTANTS ─────────────────────────────────────────────────
CONTENT_FILE  = "StreamMyMood_Content_Database.xlsx"
TRAINING_FILE = "StreamMyMood_Training_Database.xlsx"
MODEL_FILE    = "streammymood_model.pkl"

RUNTIME_RANGES = {
    "עד 30 דקות":  (0,   30),
    "30–60 דקות":  (30,  60),
    "60–120 דקות": (60, 120),
    "מעל שעתיים":  (120, 9999),
}

QUESTIONS = [
    {"id":"time_available","text":"כמה זמן פנוי יש לך לצפייה?",
     "options":["עד 30 דקות","30–60 דקות","60–120 דקות","מעל שעתיים"]},
    {"id":"watching_with","text":"עם מי את/ה נמצא?",
     "options":["לבד","בן / בת זוג","משפחה","חברים","ילדים"]},
    {"id":"mood","text":"מה המצב רוח שלך כרגע?",
     "options":[
        "קליל וחיפשתי משהו שיעשה לי טוב על הלב",
        "משועמם וחיפשתי משהו שיעורר אותי",
        "סקרן וחיפשתי סיפור להישאב אליו",
        "רגיש וחיפשתי משהו שייגע בי",
        "עמוס וחיפשתי פשוט 'להיעלם' בתוך עולם אחר",
        "מרוקן ורוצה להעביר את הזמן בלי לחשוב",
     ]},
    {"id":"review_importance","text":"עד כמה חשובות לך ביקורות?",
     "options":["חשובות מאוד!","לא באמת משנה לי"]},
    {"id":"awards_preference","text":"האם חשוב שהתוכן זכה בפרסים?",
     "options":["כן, רק זוכי פרסים","לא קריטי עבורי"]},
]

LOADING_TEXTS = [
    "מכינים המלצה שתתאים לך בול…",
    "בודקים את מצב הרוח שלך…",
    "סורקים את התכנים…",
    "מחשבים התאמה אישית במיוחד בשבילך…",
]

# ── DATA ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    content = pd.read_excel(CONTENT_FILE)

    def parse_rt(v):
        try: return int(str(v).replace("min","").replace("דקות","").strip())
        except: return None

    content["Runtime_Int"] = content["Runtime"].apply(parse_rt)
    content["Rating"] = pd.to_numeric(content["Rating"], errors="coerce")
    # Do NOT fill missing ratings with mean — leave as NaN
    content["Major_Awards_Won"] = pd.to_numeric(
        content["Major_Awards_Won"], errors="coerce").fillna(0)
    content["ID"] = content["ID"].astype(str)

    train = pd.read_excel(TRAINING_FILE)
    col_map = {
        "mood":"User_Mood","time_available":"Duration_Limit",
        "watching_with":"Watch_Group","awards_preference":"Award_Importance",
        "rating_influence":"Review_Importance","chosen_content":"Content_Title",
    }
    train.rename(columns={k:v for k,v in col_map.items() if k in train.columns}, inplace=True)

    return content, train

# ── MODEL ─────────────────────────────────────────────────────
@st.cache_resource
def load_or_train(_train_df, _content_df):
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    valid = set(_content_df["Title"].tolist())
    df    = _train_df[_train_df["Content_Title"].isin(valid)].copy()

    features = ["Duration_Limit","Watch_Group","User_Mood",
                "Review_Importance","Award_Importance"]
    encoders = {}
    X = pd.DataFrame()
    for f in features:
        le = LabelEncoder()
        X[f] = le.fit_transform(df[f].astype(str))
        encoders[f] = le

    # Target: content GENRE (learned pattern) not title
    # We merge genre from content DB to learn mood→genre mapping
    df2 = df.merge(
        _content_df[["Title","Genre"]],
        left_on="Content_Title", right_on="Title", how="left"
    )
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["Content_Title"].astype(str))

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    data = {
        "model": clf,
        "encoders": encoders,
        "le_target": le_target,
        "features": features,
        "accuracy": clf.score(X_val, y_val)
    }
    joblib.dump(data, MODEL_FILE)
    return data

# ── ENGINE ────────────────────────────────────────────────────
def get_recommendations(answers, content_df, model_data, seen_ids=None):
    seen_ids = seen_ids or set()

    mood            = answers.get("mood","")
    time_choice     = answers.get("time_available","")
    review_matters  = answers.get("review_importance","") == "חשובות מאוד!"
    awards_matter   = "כן" in answers.get("awards_preference","")

    rt_min, rt_max = RUNTIME_RANGES.get(time_choice, (0, 9999))

    # ── Dynamic weights based on user answers ──────────────────
    if review_matters:
        w_model, w_rating, w_awards = 0.50, 0.40, 0.10
    elif awards_matter:
        w_model, w_rating, w_awards = 0.55, 0.25, 0.20
    else:
        w_model, w_rating, w_awards = 0.70, 0.20, 0.10

    # ── Model probabilities ────────────────────────────────────
    answer_map = {
        "Duration_Limit":    time_choice,
        "Watch_Group":       answers.get("watching_with",""),
        "User_Mood":         mood,
        "Review_Importance": answers.get("review_importance",""),
        "Award_Importance":  answers.get("awards_preference",""),
    }
    clf      = model_data["model"]
    encoders = model_data["encoders"]
    le_target= model_data["le_target"]
    features = model_data["features"]

    row = {}
    for f in features:
        le  = encoders[f]
        val = answer_map[f]
        row[f] = le.transform([val])[0] if val in le.classes_ else 0

    proba        = clf.predict_proba(pd.DataFrame([row])[features])[0]
    known_titles = le_target.inverse_transform(clf.classes_)
    model_scores = dict(zip(known_titles, proba))

    # Max real rating (ignore NaN)
    max_rating = content_df["Rating"].dropna().max() or 10.0

    def score_content(c):
        # Model score — how likely this title given user's context
        ms = model_scores.get(c["Title"], 0.0)

        # Rating score — use actual rating, if missing use median not mean
        r = c.get("Rating")
        if pd.isna(r) or r is None:
            norm_rating = 0.5  # neutral, not inflated
        else:
            norm_rating = float(r) / max_rating

        # Awards score
        aw = min(1.0, (c.get("Major_Awards_Won", 0) or 0) / 10.0)

        return w_model * ms + w_rating * norm_rating + w_awards * aw

    # ── Filter by runtime only (no mood hard-filter) ───────────
    # The model already learned mood→content mapping from training data
    results = []
    for _, c in content_df.iterrows():
        cid = str(c["ID"])
        if cid in seen_ids:
            continue
        rt = c.get("Runtime_Int") or 0
        if not (rt_min <= rt <= rt_max):
            continue
        results.append({"row": c, "score": score_content(c), "id": cid})

    results.sort(key=lambda x: x["score"], reverse=True)

    # If fewer than 4 after runtime filter → relax runtime
    if len(results) < 4:
        seen2 = {r["id"] for r in results} | seen_ids
        extras = []
        for _, c in content_df.iterrows():
            cid = str(c["ID"])
            if cid in seen2:
                continue
            extras.append({"row": c, "score": score_content(c), "id": cid})
        extras.sort(key=lambda x: x["score"], reverse=True)
        results += extras
        if results:
            st.info("לא מצאנו תוכן מדויק לזמן שבחרת, אבל הנה התכנים שהכי מתאימים למצב הרוח שלך.")

    return results

# ── HELPERS ───────────────────────────────────────────────────
def stars(r):
    try:
        f = max(0, min(5, round(float(r) / 2)))
        return "★" * f + "☆" * (5 - f)
    except:
        return "☆☆☆☆☆"

def card_html(c):
    poster = str(c.get("Poster_URL","") or "").strip()
    if not poster or poster in ["nan","None",""]:
        poster = placeholder_src()
    ph       = placeholder_src()
    title    = c.get("Title","") or ""
    year     = int(c["Year"]) if c.get("Year") and str(c["Year"]) not in ["nan",""] else "—"
    runtime  = c.get("Runtime","—") or "—"
    rating   = c.get("Rating")
    rat_str  = f"{float(rating):.1f}" if rating and not pd.isna(rating) else "—"
    stars_str= stars(rating) if rating and not pd.isna(rating) else "☆☆☆☆☆"
    overview = c.get("Overview","") or ""
    platform = c.get("Platform","") or ""
    plat     = f'<div class="movie-platform">{platform}</div>' if platform else ""

    return f"""
<div class="movie-card">
  <img class="movie-poster" src="{poster}" alt="{title}"
       onerror="this.onerror=null;this.src='{ph}'"/>
  <div class="movie-title">{title}</div>
  <div class="movie-meta">{year} · {runtime}</div>
  {plat}
  <div class="movie-stars">{stars_str} <span style="color:rgba(255,255,255,0.55);font-size:0.78rem">{rat_str}</span></div>
  <div class="movie-overview">{overview}</div>
</div>"""

# ── SCREENS ───────────────────────────────────────────────────
def screen_welcome():
    st.markdown(logo_html(340), unsafe_allow_html=True)
    h = datetime.now().hour
    g = ("בוקר טוב" if h < 12 else "צהריים טובים" if h < 17 else
         "ערב טוב" if h < 21 else "לילה טוב")
    st.markdown(f"""
    <div class="center" style="margin-bottom:0.5rem">
      <span style="font-size:1.5rem;font-weight:800">{g}!</span>
    </div>
    <div class="center" style="margin-bottom:2.5rem">
      <span class="sub">ברוכים הבאים ל‑Stream My Mood<br>בואו נמצא לכם מה לראות.</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("לחצו כאן כדי להתחיל!"):
            st.session_state.update({
                "screen": "quiz", "q_index": 0, "answers": {},
                "seen_ids": set(), "ranked": None
            })
            st.rerun()


def screen_quiz():
    st.markdown(logo_html(240), unsafe_allow_html=True)
    q_idx = st.session_state.q_index
    q     = QUESTIONS[q_idx]
    total = len(QUESTIONS)
    pct   = int((q_idx / total) * 100)

    st.markdown(f"""
    <div class="prog-outer">
      <div class="prog-wrap">
        <div class="prog-fill" style="width:{pct}%"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div class="center" style="font-size:0.85rem;color:rgba(255,255,255,0.5);margin-bottom:0.8rem">'
        f'שאלה {q_idx+1} מתוך {total}</div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="center big" style="margin-bottom:1.4rem">{q["text"]}</div>',
        unsafe_allow_html=True)

    cur = st.session_state.answers.get(q["id"])
    idx = q["options"].index(cur) if cur in q["options"] else 0
    sel = st.radio("", q["options"], index=idx, key=f"r_{q_idx}")
    st.session_state.answers[q["id"]] = sel

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        lbl = "לשאלה הבאה" if q_idx < total - 1 else "קבלו המלצות!"
        if st.button(lbl):
            if q_idx < total - 1:
                st.session_state.q_index += 1
            else:
                st.session_state.screen = "loading"
            st.rerun()


def screen_loading():
    st.markdown(logo_html(240), unsafe_allow_html=True)
    ph = st.empty()
    for i in range(len(LOADING_TEXTS)):
        ph.markdown(f"""
        <div class="reel-wrap">
          {REEL_SVG}
          <div class="loading-txt">{LOADING_TEXTS[i]}</div>
        </div>""", unsafe_allow_html=True)
        time.sleep(0.7)
    st.session_state.screen = "results"
    st.rerun()


def screen_results(content_df, model_data):
    st.markdown(logo_html(240), unsafe_allow_html=True)
    answers  = st.session_state.answers
    seen_ids = st.session_state.get("seen_ids", set())

    if not st.session_state.get("ranked"):
        st.session_state.ranked = get_recommendations(
            answers, content_df, model_data, set())

    ranked    = st.session_state.ranked
    remaining = [r for r in ranked if r["id"] not in seen_ids]
    batch     = remaining[:4]

    if not batch:
        st.markdown(
            '<div class="center big" style="margin:2rem 0">לא קיימות המלצות נוספות בהתאם להגדרות שלך.</div>',
            unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("התחל מחדש"):
                for k in ["screen","q_index","answers","seen_ids","ranked"]:
                    st.session_state.pop(k, None)
                st.rerun()
        return

    st.markdown(
        '<div class="center" style="font-size:1.25rem;font-weight:800;margin-bottom:1.4rem">ההמלצות שלנו עבורך</div>',
        unsafe_allow_html=True)

    cols = st.columns(4)
    for col, r in zip(cols, batch):
        with col:
            st.markdown(card_html(r["row"]), unsafe_allow_html=True)
        seen_ids.add(r["id"])
    st.session_state.seen_ids = seen_ids

    st.markdown("<br><br>", unsafe_allow_html=True)
    next_avail = [r for r in ranked if r["id"] not in seen_ids]

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if next_avail:
            if st.button("לא אהבתי את ההמלצות האלה, אשמח להמלצות נוספות"):
                st.rerun()
        else:
            st.markdown(
                '<div class="center sub">לא קיימות המלצות נוספות בהתאם להגדרות שלך.</div>',
                unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("התחל מחדש"):
            for k in ["screen","q_index","answers","seen_ids","ranked"]:
                st.session_state.pop(k, None)
            st.rerun()


# ── MAIN ──────────────────────────────────────────────────────
def main():
    if "screen" not in st.session_state:
        st.session_state.screen = "welcome"

    content_df, train_df = load_data()
    model_data = load_or_train(train_df, content_df)

    s = st.session_state.screen
    if   s == "welcome": screen_welcome()
    elif s == "quiz":    screen_quiz()
    elif s == "loading": screen_loading()
    elif s == "results": screen_results(content_df, model_data)

if __name__ == "__main__":
    main()
