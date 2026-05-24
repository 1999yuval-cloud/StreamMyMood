# ============================================================
#  StreamMyMood — app.py  v3
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

# ── LOGO ──────────────────────────────────────────────────────
def get_logo_b64():
    if os.path.exists("logo.png"):
        with open("logo.png","rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = get_logo_b64()

def logo_html(width=280):
    if LOGO_B64:
        return (f'<div style="text-align:center;margin-bottom:1.2rem">'
                f'<img src="data:image/png;base64,{LOGO_B64}" '
                f'width="{width}" style="mix-blend-mode:screen;filter:brightness(1.1)"/>'
                f'</div>')
    return '<div style="text-align:center;font-size:2rem;font-weight:800;color:#E91E63;margin-bottom:1.2rem">StreamMyMood</div>'

def placeholder_src():
    if LOGO_B64:
        return f"data:image/png;base64,{LOGO_B64}"
    return "https://via.placeholder.com/300x420/2B0015/ffffff?text=StreamMyMood"

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;600;700;800&display=swap');

* {{ font-family: 'Heebo', 'Segoe UI', Arial, sans-serif !important; }}

/* Background */
.stApp, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(ellipse at top, #3a0020 0%, #2B0015 40%, #1a000d 100%) !important;
  color: white !important;
  min-height: 100vh;
}}

/* Film grain overlay */
.stApp::before {{
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  opacity: 0.06;
  pointer-events: none;
  z-index: 0;
}}

/* Bokeh lights */
.stApp::after {{
  content: '';
  position: fixed;
  top: -100px; left: -100px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(180,0,60,0.08) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}}

[data-testid="stHeader"] {{ background: transparent !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}

.block-container {{
  padding-top: 2rem !important;
  max-width: 1150px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}}

h1,h2,h3,p,label,div,span {{
  color: white !important;
}}

/* ── Centered buttons ── */
.stButton {{
  display: flex !important;
  justify-content: center !important;
}}
.stButton > button {{
  background: linear-gradient(135deg, #7a0025 0%, #b0002e 100%) !important;
  color: white !important;
  border: 1.5px solid rgba(255,255,255,0.25) !important;
  border-radius: 30px !important;
  padding: 0.7rem 2.5rem !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  font-family: 'Heebo', sans-serif !important;
  min-width: 240px !important;
  letter-spacing: 0.3px;
  transition: all 0.2s !important;
  box-shadow: 0 4px 20px rgba(180,0,50,0.3) !important;
}}
.stButton > button:hover {{
  background: linear-gradient(135deg, #b0002e 0%, #d4003a 100%) !important;
  border-color: rgba(255,255,255,0.5) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 25px rgba(180,0,50,0.5) !important;
}}

/* ── Radio buttons ── */
div[data-testid="stRadio"] > label {{ display: none !important; }}
div[data-testid="stRadio"] > div {{
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  gap: 0.5rem !important;
}}
div[data-testid="stRadio"] label {{
  background: rgba(255,255,255,0.07) !important;
  border: 1.5px solid rgba(255,255,255,0.2) !important;
  border-radius: 14px !important;
  padding: 0.85rem 2rem !important;
  width: 500px !important;
  max-width: 92vw !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  text-align: center !important;
  font-size: 1rem !important;
  font-weight: 500 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}}
div[data-testid="stRadio"] label:hover {{
  background: rgba(180,0,50,0.3) !important;
  border-color: rgba(255,100,100,0.6) !important;
}}
div[data-testid="stRadio"] label[data-checked="true"] {{
  background: rgba(180,0,50,0.5) !important;
  border-color: rgba(255,120,120,0.8) !important;
  font-weight: 700 !important;
}}

/* ── Progress bar ── */
.prog-wrap {{
  background: rgba(255,255,255,0.1);
  border-radius: 10px; height: 6px;
  margin: 0.3rem auto 1.8rem auto;
  max-width: 560px;
}}
.prog-fill {{
  background: linear-gradient(90deg, #b0002e, #ff4d6d);
  height: 6px; border-radius: 10px;
  transition: width 0.4s ease;
  box-shadow: 0 0 8px rgba(255,77,109,0.6);
}}

/* ── Movie cards ── */
.card-grid {{ display: flex; gap: 1rem; justify-content: center; }}
.movie-card {{
  background: rgba(0,0,0,0.45);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 1rem;
  backdrop-filter: blur(6px);
  transition: transform 0.2s, box-shadow 0.2s;
  flex: 1;
}}
.movie-card:hover {{
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(180,0,50,0.3);
}}
.movie-poster {{
  width: 100%; border-radius: 10px;
  margin-bottom: 0.7rem;
  object-fit: cover; height: 290px;
  background: #1a000d;
}}
.movie-title {{
  font-size: 1.05rem !important;
  font-weight: 800 !important;
  margin-bottom: 0.2rem !important;
  line-height: 1.3;
}}
.movie-meta {{
  font-size: 0.8rem !important;
  color: rgba(255,255,255,0.6) !important;
  margin-bottom: 0.35rem !important;
  direction: ltr;
  text-align: left;
}}
.movie-platform {{
  font-size: 0.78rem !important;
  color: #ff9ab0 !important;
  margin-bottom: 0.35rem !important;
  font-weight: 600 !important;
}}
.movie-stars {{ color: #FFD700; font-size: 0.88rem; margin-bottom: 0.4rem; }}
.movie-overview {{
  font-size: 0.82rem !important;
  color: rgba(255,255,255,0.82) !important;
  height: 100px; overflow-y: auto;
  line-height: 1.55; padding-right: 2px;
}}

/* ── Loading clapper ── */
.clapper-wrap {{ text-align:center; padding: 3rem 0; }}
.clapper-top {{
  display: inline-block;
  font-size: 3.5rem;
  transform-origin: bottom left;
  animation: clap-top 0.8s ease-in-out infinite alternate;
}}
.clapper-body {{ font-size: 3.5rem; display: inline-block; }}
@keyframes clap-top {{
  0%   {{ transform: rotate(0deg); }}
  100% {{ transform: rotate(-45deg); }}
}}
.loading-txt {{
  font-size: 1.15rem !important;
  color: rgba(255,255,255,0.85) !important;
  margin-top: 1.2rem !important;
  font-weight: 500 !important;
}}

.center {{ text-align: center !important; }}
.big {{ font-size: 1.35rem !important; font-weight: 700 !important; }}
.sub {{ font-size: 1rem !important; color: rgba(255,255,255,0.75) !important; line-height: 1.6; }}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
CONTENT_FILE  = "StreamMyMood_Content_Database.xlsx"
TRAINING_FILE = "StreamMyMood_Training_Database.xlsx"
MODEL_FILE    = "streammymood_model.pkl"

# Runtime range per time choice (min, max) in minutes
# Each content item's runtime must FALL WITHIN this range
RUNTIME_RANGES = {
    "עד 30 דקות":   (0,   30),
    "30–60 דקות":   (30,  60),
    "60–120 דקות":  (60,  120),
    "מעל שעתיים":   (120, 9999),
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
    content["Rating"]      = pd.to_numeric(content["Rating"], errors="coerce")
    content["Rating"].fillna(content["Rating"].mean(), inplace=True)
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

    # Build mood → valid titles map (from real training data)
    mood_titles = (train.groupby("User_Mood")["Content_Title"]
                   .apply(set).to_dict())

    return content, train, mood_titles

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
        le   = LabelEncoder()
        X[f] = le.fit_transform(df[f].astype(str))
        encoders[f] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(df["Content_Title"].astype(str))

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                  min_samples_leaf=1, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    data = {"model":clf,"encoders":encoders,"le_target":le_target,
            "features":features,"accuracy":clf.score(X_val, y_val)}
    joblib.dump(data, MODEL_FILE)
    return data

# ── RECOMMENDATION ENGINE ─────────────────────────────────────
def get_recommendations(answers, content_df, model_data, mood_titles, seen_ids=None):
    seen_ids = seen_ids or set()
    clf      = model_data["model"]
    encoders = model_data["encoders"]
    le_target= model_data["le_target"]
    features = model_data["features"]

    mood          = answers.get("mood","")
    time_choice   = answers.get("time_available","")
    wants_awards  = "כן" in answers.get("awards_preference","")

    # Step 1 — mood filter: only titles that appear under this mood in training data
    valid_for_mood = mood_titles.get(mood, set())

    # Step 2 — runtime filter: content runtime must fall in the chosen time range
    rt_min, rt_max = RUNTIME_RANGES.get(time_choice, (0, 9999))

    # Step 3 — score with model
    answer_map = {
        "Duration_Limit":    time_choice,
        "Watch_Group":       answers.get("watching_with",""),
        "User_Mood":         mood,
        "Review_Importance": answers.get("review_importance",""),
        "Award_Importance":  answers.get("awards_preference",""),
    }
    row = {}
    for f in features:
        le  = encoders[f]
        val = answer_map[f]
        row[f] = le.transform([val])[0] if val in le.classes_ else 0

    X_in  = pd.DataFrame([row])[features]
    proba = clf.predict_proba(X_in)[0]
    titles_known = le_target.inverse_transform(clf.classes_)
    model_scores = dict(zip(titles_known, proba))

    max_rating = content_df["Rating"].max() or 10.0

    results = []
    for _, c in content_df.iterrows():
        cid   = str(c["ID"])
        title = c["Title"]
        if cid in seen_ids:
            continue

        # Mood filter — hard
        if title not in valid_for_mood:
            continue

        # Runtime filter — hard
        rt = c.get("Runtime_Int") or 0
        if not (rt_min <= rt <= rt_max):
            continue

        ms           = model_scores.get(title, 0.0)
        norm_rating  = c["Rating"] / max_rating
        awards_won   = c.get("Major_Awards_Won", 0) or 0
        awards_score = min(1.0, awards_won / 10.0)

        final = 0.6 * ms + 0.3 * norm_rating + 0.1 * awards_score

        if wants_awards and awards_won > 0:
            final *= 1.25

        results.append({"row":c, "score":final, "id":cid})

    results.sort(key=lambda x: x["score"], reverse=True)

    # If fewer than 4 — supplement without mood filter but keep runtime filter
    if len(results) < 4:
        seen_in_results = {r["id"] for r in results}
        extras = []
        for _, c in content_df.iterrows():
            cid = str(c["ID"])
            if cid in seen_ids or cid in seen_in_results:
                continue
            rt = c.get("Runtime_Int") or 0
            if not (rt_min <= rt <= rt_max):
                continue
            ms           = model_scores.get(c["Title"], 0.0)
            norm_rating  = c["Rating"] / max_rating
            awards_won   = c.get("Major_Awards_Won", 0) or 0
            awards_score = min(1.0, awards_won / 10.0)
            final        = 0.6*ms + 0.3*norm_rating + 0.1*awards_score
            extras.append({"row":c,"score":final,"id":cid})
        extras.sort(key=lambda x: x["score"], reverse=True)
        results += extras
        if results:
            st.info("לא מצאנו תוכן מדויק, אבל הנה התכנים הכי קרובים לבקשה שלך.")

    return results

# ── HELPERS ───────────────────────────────────────────────────
def stars(r):
    f = max(0, min(5, round(float(r)/2)))
    return "★"*f + "☆"*(5-f)

def card_html(c):
    poster = str(c.get("Poster_URL","") or "").strip()
    if not poster or poster in ["nan","None",""]:
        poster = placeholder_src()

    title    = c.get("Title","") or ""
    year     = int(c["Year"]) if c.get("Year") and str(c["Year"]) != "nan" else "—"
    runtime  = c.get("Runtime","—") or "—"
    rating   = float(c.get("Rating",7.9) or 7.9)
    overview = c.get("Overview","") or ""
    platform = c.get("Platform","") or ""
    ph       = placeholder_src()

    return f"""
<div class="movie-card">
  <img class="movie-poster" src="{poster}" alt="{title}"
       onerror="this.onerror=null;this.src='{ph}'"/>
  <div class="movie-title">{title}</div>
  <div class="movie-meta">{year} &nbsp;·&nbsp; {runtime}</div>
  {'<div class="movie-platform">'+platform+'</div>' if platform else ''}
  <div class="movie-stars">{stars(rating)}&nbsp;<span style="color:rgba(255,255,255,0.6);font-size:0.78rem">{rating:.1f}</span></div>
  <div class="movie-overview">{overview}</div>
</div>"""

# ── SCREENS ───────────────────────────────────────────────────
def screen_welcome():
    st.markdown(logo_html(300), unsafe_allow_html=True)

    h = datetime.now().hour
    g = ("בוקר טוב" if h < 12 else
         "צהריים טובים" if h < 17 else
         "ערב טוב" if h < 21 else "לילה טוב")

    st.markdown(f"""
    <div class="center" style="margin-bottom:0.6rem">
      <span style="font-size:1.4rem;font-weight:700">{g}!</span>
    </div>
    <div class="center" style="margin-bottom:2.5rem">
      <span class="sub">ברוכים הבאים ל‑Stream My Mood<br>בואו נמצא לכם מה לראות.</span>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if st.button("לחצו כאן כדי להתחיל!"):
            st.session_state.update({
                "screen":"quiz","q_index":0,"answers":{},
                "seen_ids":set(),"ranked":None
            })
            st.rerun()


def screen_quiz():
    st.markdown(logo_html(200), unsafe_allow_html=True)
    q_idx = st.session_state.q_index
    q     = QUESTIONS[q_idx]
    total = len(QUESTIONS)
    pct   = int((q_idx/total)*100)

    st.markdown(f'<div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="center" style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-bottom:0.8rem">שאלה {q_idx+1} מתוך {total}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="center big" style="margin-bottom:1.4rem">{q["text"]}</div>',
                unsafe_allow_html=True)

    cur = st.session_state.answers.get(q["id"])
    idx = q["options"].index(cur) if cur in q["options"] else 0
    sel = st.radio("", q["options"], index=idx, key=f"r_{q_idx}")
    st.session_state.answers[q["id"]] = sel

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        lbl = "לשאלה הבאה" if q_idx < total-1 else "קבלו המלצות!"
        if st.button(lbl):
            if q_idx < total-1:
                st.session_state.q_index += 1
            else:
                st.session_state.screen = "loading"
            st.rerun()


def screen_loading():
    st.markdown(logo_html(200), unsafe_allow_html=True)
    ph = st.empty()
    for i in range(len(LOADING_TEXTS)):
        txt = LOADING_TEXTS[i]
        ph.markdown(f"""
        <div class="clapper-wrap">
          <span class="clapper-top">🎬</span>
          <div class="loading-txt">{txt}</div>
        </div>""", unsafe_allow_html=True)
        time.sleep(0.7)
    st.session_state.screen = "results"
    st.rerun()


def screen_results(content_df, model_data, mood_titles):
    st.markdown(logo_html(200), unsafe_allow_html=True)

    answers  = st.session_state.answers
    seen_ids = st.session_state.get("seen_ids", set())

    if not st.session_state.get("ranked"):
        st.session_state.ranked = get_recommendations(
            answers, content_df, model_data, mood_titles, set())

    ranked    = st.session_state.ranked
    remaining = [r for r in ranked if r["id"] not in seen_ids]
    batch     = remaining[:4]

    if not batch:
        st.markdown('<div class="center big" style="margin:2rem 0">לא קיימות המלצות נוספות בהתאם להגדרות שלך.</div>',
                    unsafe_allow_html=True)
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            if st.button("התחל מחדש"):
                for k in ["screen","q_index","answers","seen_ids","ranked"]:
                    st.session_state.pop(k, None)
                st.rerun()
        return

    st.markdown('<div class="center" style="font-size:1.25rem;font-weight:700;margin-bottom:1.4rem">ההמלצות שלנו עבורך</div>',
                unsafe_allow_html=True)

    cols = st.columns(4)
    for col, r in zip(cols, batch):
        with col:
            st.markdown(card_html(r["row"]), unsafe_allow_html=True)
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
            st.markdown('<div class="center sub">לא קיימות המלצות נוספות בהתאם להגדרות שלך.</div>',
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

    content_df, train_df, mood_titles = load_data()
    model_data = load_or_train(train_df, content_df)

    s = st.session_state.screen
    if   s == "welcome": screen_welcome()
    elif s == "quiz":    screen_quiz()
    elif s == "loading": screen_loading()
    elif s == "results": screen_results(content_df, model_data, mood_titles)

if __name__ == "__main__":
    main()
