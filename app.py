# ============================================================
#  StreamMyMood — app.py  v4
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

def get_logo_b64():
    if os.path.exists("logo - StreamMyMood.png"):
        with open("logo - StreamMyMood.png","rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = get_logo_b64()

def logo_html(width=280):
    if LOGO_B64:
        return (f'<div style="text-align:center;margin-bottom:1.2rem">'
                f'<img src="data:image/png;base64,{LOGO_B64}" width="{width}"/>'
                f'</div>')
    return '<div style="text-align:center;font-size:2.5rem;font-weight:900;color:#ff4d6d;margin-bottom:1.2rem">StreamMyMood</div>'

def placeholder_src():
    if LOGO_B64:
        return f"data:image/png;base64,{LOGO_B64}"
    return "https://via.placeholder.com/300x420/3d0018/ffffff?text=StreamMyMood"

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;600;700;800&display=swap');

html, body, * {
  font-family: 'Assistant', Arial, sans-serif !important;
  direction: rtl !important;
}

/* ── Background: true dark burgundy gradient ── */
.stApp, [data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg,
    #1a0008 0%,
    #2d0012 30%,
    #3d0018 60%,
    #4a0020 100%) !important;
  min-height: 100vh;
  color: white !important;
}
[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
  padding-top: 2rem !important;
  max-width: 1150px;
  margin: 0 auto;
}

h1,h2,h3,p,label,div,span { color: white !important; }

/* ── Film grain ── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23g)' opacity='0.035'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 0; opacity: 0.5;
}

/* bokeh corners */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed;
  top: -150px; right: -150px;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(200,0,60,0.07) 0%, transparent 70%);
  pointer-events: none; z-index: 0;
}

/* ── All buttons centered ── */
div[data-testid="stVerticalBlock"] > div {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.stButton {
  display: flex !important;
  justify-content: center !important;
  width: 100% !important;
}
.stButton > button {
  background: linear-gradient(135deg, #7a0025, #b30030) !important;
  color: white !important;
  border: 1.5px solid rgba(255,255,255,0.2) !important;
  border-radius: 30px !important;
  padding: 0.75rem 2.8rem !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  font-family: 'Assistant', sans-serif !important;
  min-width: 260px !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 18px rgba(180,0,50,0.35) !important;
  direction: rtl !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #b30030, #d4003a) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px rgba(180,0,50,0.5) !important;
}

/* ── Radio ── */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  gap: 0.55rem !important;
  width: 100% !important;
}
div[data-testid="stRadio"] label {
  background: rgba(255,255,255,0.06) !important;
  border: 1.5px solid rgba(255,255,255,0.18) !important;
  border-radius: 14px !important;
  padding: 0.85rem 2rem !important;
  width: 520px !important;
  max-width: 90vw !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  text-align: center !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  direction: rtl !important;
}
div[data-testid="stRadio"] label:hover {
  background: rgba(180,0,50,0.25) !important;
  border-color: rgba(255,100,130,0.55) !important;
}
div[data-testid="stRadio"] label[data-checked="true"] {
  background: rgba(180,0,50,0.45) !important;
  border-color: rgba(255,120,150,0.8) !important;
  font-weight: 800 !important;
}

/* ── Progress bar ── */
.prog-wrap {
  background: rgba(255,255,255,0.1);
  border-radius: 10px; height: 6px;
  width: 560px; max-width: 90vw;
  margin: 0.3rem auto 1.8rem auto;
}
.prog-fill {
  background: linear-gradient(90deg, #b30030, #ff4d6d);
  height: 6px; border-radius: 10px;
  transition: width 0.4s ease;
  box-shadow: 0 0 8px rgba(255,77,109,0.5);
}

/* ── Movie cards ── */
.movie-card {
  background: rgba(0,0,0,0.45);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 1rem;
  backdrop-filter: blur(6px);
  transition: transform 0.2s, box-shadow 0.2s;
  text-align: right;
  direction: rtl;
}
.movie-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 28px rgba(180,0,50,0.3);
}
.movie-poster {
  width: 100%; border-radius: 10px;
  margin-bottom: 0.7rem;
  object-fit: cover; height: 280px;
  background: #1a0008;
  display: block;
}
.movie-title {
  font-size: 1rem !important;
  font-weight: 800 !important;
  margin-bottom: 0.2rem !important;
  line-height: 1.3 !important;
  color: white !important;
}
.movie-meta {
  font-size: 0.8rem !important;
  color: rgba(255,255,255,0.6) !important;
  margin-bottom: 0.25rem !important;
}
.movie-platform {
  font-size: 0.8rem !important;
  color: #ff9ab0 !important;
  font-weight: 700 !important;
  margin-bottom: 0.3rem !important;
}
.movie-stars {
  color: #FFD700;
  font-size: 0.88rem;
  margin-bottom: 0.4rem;
}
.movie-overview {
  font-size: 0.82rem !important;
  color: rgba(255,255,255,0.8) !important;
  height: 100px; overflow-y: auto;
  line-height: 1.55;
}

/* ── Loading ── */
.clapper-wrap { text-align: center; padding: 2.5rem 0; }
.clapper-icon {
  font-size: 3.5rem;
  display: inline-block;
  animation: clap 0.7s ease-in-out infinite alternate;
}
@keyframes clap {
  0%   { transform: rotate(-20deg); }
  100% { transform: rotate(10deg) scale(1.15); }
}
.loading-txt {
  font-size: 1.15rem !important;
  color: rgba(255,255,255,0.85) !important;
  margin-top: 1.2rem !important;
  font-weight: 600 !important;
}

/* ── Helpers ── */
.center { text-align: center !important; }
.big { font-size: 1.35rem !important; font-weight: 800 !important; }
.sub { font-size: 1rem !important; color: rgba(255,255,255,0.75) !important; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
CONTENT_FILE  = "StreamMyMood_Content_Database.xlsx"
TRAINING_FILE = "StreamMyMood_Training_Database.xlsx"
MODEL_FILE    = "streammymood_model.pkl"

RUNTIME_RANGES = {
    "עד 30 דקות":   (0,   30),
    "30–60 דקות":   (30,  60),
    "60–120 דקות":  (60, 120),
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
    content["Rating"] = pd.to_numeric(content["Rating"], errors="coerce")
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

    mood_titles = train.groupby("User_Mood")["Content_Title"].apply(set).to_dict()
    return content, train, mood_titles

# ── MODEL ─────────────────────────────────────────────────────
@st.cache_resource
def load_or_train(_train_df, _content_df):
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    valid = set(_content_df["Title"].tolist())
    df    = _train_df[_train_df["Content_Title"].isin(valid)].copy()

    features = ["Duration_Limit","Watch_Group","User_Mood","Review_Importance","Award_Importance"]
    encoders = {}
    X = pd.DataFrame()
    for f in features:
        le = LabelEncoder()
        X[f] = le.fit_transform(df[f].astype(str))
        encoders[f] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(df["Content_Title"].astype(str))

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    data = {"model":clf,"encoders":encoders,"le_target":le_target,
            "features":features,"accuracy":clf.score(X_val,y_val)}
    joblib.dump(data, MODEL_FILE)
    return data

# ── ENGINE ────────────────────────────────────────────────────
def get_recommendations(answers, content_df, model_data, mood_titles, seen_ids=None):
    seen_ids = seen_ids or set()
    mood         = answers.get("mood","")
    time_choice  = answers.get("time_available","")
    wants_awards = "כן" in answers.get("awards_preference","")
    rt_min, rt_max = RUNTIME_RANGES.get(time_choice,(0,9999))

    valid_for_mood = mood_titles.get(mood, set())

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

    proba  = clf.predict_proba(pd.DataFrame([row])[features])[0]
    titles_known = le_target.inverse_transform(clf.classes_)
    model_scores = dict(zip(titles_known, proba))
    max_rating   = content_df["Rating"].max() or 10.0

    def score(c):
        ms          = model_scores.get(c["Title"],0.0)
        norm_rating = c["Rating"] / max_rating
        aw          = min(1.0,(c.get("Major_Awards_Won",0) or 0)/10.0)
        f           = 0.6*ms + 0.3*norm_rating + 0.1*aw
        if wants_awards and (c.get("Major_Awards_Won",0) or 0) > 0:
            f *= 1.25
        return f

    # Pass 1 — mood filter + runtime filter
    results = []
    for _, c in content_df.iterrows():
        cid = str(c["ID"])
        if cid in seen_ids: continue
        if c["Title"] not in valid_for_mood: continue
        rt = c.get("Runtime_Int") or 0
        if not (rt_min <= rt <= rt_max): continue
        results.append({"row":c,"score":score(c),"id":cid})

    results.sort(key=lambda x: x["score"], reverse=True)

    # Pass 2 — if < 4, relax mood filter, keep runtime
    if len(results) < 4:
        seen2 = {r["id"] for r in results} | seen_ids
        extras = []
        for _, c in content_df.iterrows():
            cid = str(c["ID"])
            if cid in seen2: continue
            rt = c.get("Runtime_Int") or 0
            if not (rt_min <= rt <= rt_max): continue
            extras.append({"row":c,"score":score(c),"id":cid})
        extras.sort(key=lambda x: x["score"], reverse=True)
        results += extras
        if results:
            st.info("לא מצאנו תוכן מדויק, אבל הנה התכנים הכי קרובים לבקשה שלך.")

    return results

# ── HELPERS ───────────────────────────────────────────────────
def stars(r):
    f = max(0,min(5,round(float(r)/2)))
    return "★"*f + "☆"*(5-f)

def card_html(c):
    poster  = str(c.get("Poster_URL","") or "").strip()
    if not poster or poster in ["nan","None",""]:
        poster = placeholder_src()
    ph       = placeholder_src()
    title    = c.get("Title","") or ""
    year     = int(c["Year"]) if c.get("Year") and str(c["Year"]) not in ["nan",""] else "—"
    runtime  = c.get("Runtime","—") or "—"
    rating   = float(c.get("Rating",7.9) or 7.9)
    overview = c.get("Overview","") or ""
    platform = c.get("Platform","") or ""

    plat_html = f'<div class="movie-platform">{platform}</div>' if platform else ""

    return f"""
<div class="movie-card">
  <img class="movie-poster" src="{poster}" alt="{title}"
       onerror="this.onerror=null;this.src='{ph}'"/>
  <div class="movie-title">{title}</div>
  <div class="movie-meta">{year} · {runtime}</div>
  {plat_html}
  <div class="movie-stars">{stars(rating)} <span style="color:rgba(255,255,255,0.55);font-size:0.78rem">{rating:.1f}</span></div>
  <div class="movie-overview">{overview}</div>
</div>"""

# ── SCREENS ───────────────────────────────────────────────────
def screen_welcome():
    st.markdown(logo_html(300), unsafe_allow_html=True)

    h = datetime.now().hour
    g = ("בוקר טוב" if h<12 else "צהריים טובים" if h<17 else
         "ערב טוב" if h<21 else "לילה טוב")

    st.markdown(f"""
    <div class="center" style="margin-bottom:0.5rem">
      <span style="font-size:1.4rem;font-weight:800">{g}!</span>
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

    st.markdown(
        f'<div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="center" style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-bottom:0.8rem">שאלה {q_idx+1} מתוך {total}</div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="center big" style="margin-bottom:1.4rem">{q["text"]}</div>',
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
        ph.markdown(f"""
        <div class="clapper-wrap">
          <span class="clapper-icon">🎬</span>
          <div class="loading-txt">{LOADING_TEXTS[i]}</div>
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

    st.markdown('<div class="center" style="font-size:1.25rem;font-weight:800;margin-bottom:1.4rem">ההמלצות שלנו עבורך</div>',
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
