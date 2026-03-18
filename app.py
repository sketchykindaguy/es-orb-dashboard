"""
ES Futures — ORB Probability Engine
Opening Range Breakout + Pullback Analysis Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import time, datetime
import os

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ES ORB Probability Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { background-color: #0a0e17; }
    .main-header {
        background: linear-gradient(135deg, #0d1320 0%, #131b2e 100%);
        border: 1px solid #1e2a42; border-radius: 12px;
        padding: 24px 32px; margin-bottom: 20px; text-align: center;
    }
    .main-header h1 {
        font-family: 'JetBrains Mono', monospace; font-size: 1.6rem;
        font-weight: 700; color: #e2e8f0; margin: 0; letter-spacing: 2px;
    }
    .main-header p {
        font-family: 'Inter', sans-serif; color: #64748b;
        font-size: 0.85rem; margin: 6px 0 0 0;
    }
    .stat-card {
        background: linear-gradient(180deg, #111827 0%, #0d1117 100%);
        border: 1px solid #1e2a42; border-radius: 10px;
        padding: 18px; text-align: center;
    }
    .stat-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
        font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 2px;
        margin-bottom: 6px;
    }
    .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; margin: 4px 0; }
    .stat-sub { font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #475569; }
    .green { color: #22c55e; } .red { color: #ef4444; }
    .blue { color: #3b82f6; } .amber { color: #f59e0b; }
    .purple { color: #8b5cf6; } .white { color: #e2e8f0; }
    .section-divider { border-top: 1px solid #1a2236; margin: 20px 0; }
    .panel {
        background: linear-gradient(180deg, #0f1623 0%, #0d1219 100%);
        border: 1px solid #1e2a42; border-radius: 10px; padding: 16px 20px;
    }
    .panel-title {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        font-weight: 600; color: #3b82f6; text-transform: uppercase;
        letter-spacing: 3px; margin-bottom: 12px; padding-bottom: 6px;
        border-bottom: 1px solid #1a2236;
    }
    div[data-baseweb="select"] > div {
        background-color: #111827 !important; border-color: #1e2a42 !important;
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important;
        font-weight: 600 !important; color: #94a3b8 !important;
        text-transform: uppercase !important; letter-spacing: 1px !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1219; border-right: 1px solid #1e2a42;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'JetBrains Mono', monospace; color: #3b82f6;
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading ES futures data...")
def load_data():
    # Look for parquet in same directory as app
    app_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(app_dir, "data", "ES_precomputed.parquet")
    if not os.path.exists(parquet_path):
        # Fallback: look in app root
        parquet_path = os.path.join(app_dir, "ES_precomputed.parquet")
    
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    elif str(df.index.tz) != "US/Eastern":
        df.index = df.index.tz_convert("US/Eastern")
    df.index.name = "timestamp"
    
    df["trade_date"] = df.index.date
    mask_evening = df.index.hour >= 18
    evening_dates = pd.to_datetime(df.loc[mask_evening, "trade_date"]) + pd.Timedelta(days=1)
    df.loc[mask_evening, "trade_date"] = evening_dates.dt.date.values
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["dow"] = df["trade_date"].dt.dayofweek
    df.loc[df["dow"] == 6, "dow"] = 0
    df["bar_time"] = df.index.time
    return df

df = load_data()


# ══════════════════════════════════════════════════════════════
#  ORB ENGINE
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Computing ORB probabilities...")
def compute_orb(_df, dr_start, dr_end, session_end, target_pts, day_filter, date_range):
    max_date = _df["trade_date"].max()
    range_map = {
        "Entire Dataset": None, "Last 10 Years": pd.DateOffset(years=10),
        "Last 5 Years": pd.DateOffset(years=5), "Last 1 Year": pd.DateOffset(years=1),
        "Last 6 Months": pd.DateOffset(months=6), "Last 3 Months": pd.DateOffset(months=3),
    }
    offset = range_map.get(date_range)
    data = _df[_df["trade_date"] >= (max_date - offset)].copy() if offset else _df.copy()
    
    day_map = {"Mondays": 0, "Tuesdays": 1, "Wednesdays": 2, "Thursdays": 3, "Fridays": 4}
    if day_filter != "All Days":
        data = data[data["dow"] == day_map[day_filter]]
    
    results = []
    for td, group in data.groupby("trade_date"):
        dr_bars = group[(group["bar_time"] >= dr_start) & (group["bar_time"] < dr_end)]
        if len(dr_bars) < 2:
            continue
        dr_high, dr_low = dr_bars["high"].max(), dr_bars["low"].min()
        
        session = group[(group["bar_time"] >= dr_end) & (group["bar_time"] <= session_end)]
        if session.empty:
            continue
        
        up_target, dn_target = dr_high + target_pts, dr_low - target_pts
        upside_hit = session["high"].max() >= up_target
        downside_hit = session["low"].min() <= dn_target
        
        dr_close_time = dr_bars.index[-1]
        up_min = dn_min = None
        if upside_hit:
            hit = session[session["high"] >= up_target].index[0]
            up_min = (hit - dr_close_time).total_seconds() / 60
        if downside_hit:
            hit = session[session["low"] <= dn_target].index[0]
            dn_min = (hit - dr_close_time).total_seconds() / 60
        
        results.append({
            "trade_date": td, "dow": group["dow"].iloc[0],
            "dr_high": dr_high, "dr_low": dr_low, "dr_range": dr_high - dr_low,
            "upside_hit": upside_hit, "downside_hit": downside_hit,
            "either_hit": upside_hit or downside_hit,
            "both_hit": upside_hit and downside_hit,
            "up_minutes": up_min, "dn_minutes": dn_min,
        })
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
#  PULLBACK ENGINE
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Analyzing pullback patterns...")
def analyze_pullbacks(_df, dr_start, dr_end, session_end, target_pts, day_filter, date_range):
    max_date = _df["trade_date"].max()
    range_map = {
        "Entire Dataset": None, "Last 10 Years": pd.DateOffset(years=10),
        "Last 5 Years": pd.DateOffset(years=5), "Last 1 Year": pd.DateOffset(years=1),
        "Last 6 Months": pd.DateOffset(months=6), "Last 3 Months": pd.DateOffset(months=3),
    }
    offset = range_map.get(date_range)
    data = _df[_df["trade_date"] >= (max_date - offset)].copy() if offset else _df.copy()
    
    day_map = {"Mondays": 0, "Tuesdays": 1, "Wednesdays": 2, "Thursdays": 3, "Fridays": 4}
    if day_filter != "All Days":
        data = data[data["dow"] == day_map[day_filter]]
    
    results = []
    for td, group in data.groupby("trade_date"):
        dr_bars = group[(group["bar_time"] >= dr_start) & (group["bar_time"] < dr_end)]
        if len(dr_bars) < 2:
            continue
        dr_high, dr_low = dr_bars["high"].max(), dr_bars["low"].min()
        
        session = group[(group["bar_time"] >= dr_end) & (group["bar_time"] <= session_end)]
        if session.empty:
            continue
        
        up_target, dn_target = dr_high + target_pts, dr_low - target_pts
        
        up_pullbacks, dn_pullbacks = [], []
        up_state, dn_state = "inside", "inside"
        up_hit, dn_hit = False, False
        up_time, dn_time = None, None
        up_bo_time, dn_bo_time = None, None
        first_up_bo, first_dn_bo = None, None
        
        for idx, bar in session.iterrows():
            c, h, l = bar["close"], bar["high"], bar["low"]
            
            if not up_hit:
                if h >= up_target:
                    up_hit, up_time = True, idx
                if up_state == "inside" and c > dr_high:
                    up_state, up_bo_time = "above", idx
                    if first_up_bo is None: first_up_bo = idx
                elif up_state == "above" and c <= dr_high and c >= dr_low:
                    up_state = "inside"
                    up_pullbacks.append({"breakout_time": up_bo_time, "pullback_time": idx})
            
            if not dn_hit:
                if l <= dn_target:
                    dn_hit, dn_time = True, idx
                if dn_state == "inside" and c < dr_low:
                    dn_state, dn_bo_time = "below", idx
                    if first_dn_bo is None: first_dn_bo = idx
                elif dn_state == "below" and c >= dr_low and c <= dr_high:
                    dn_state = "inside"
                    dn_pullbacks.append({"breakout_time": dn_bo_time, "pullback_time": idx})
        
        results.append({
            "trade_date": td, "dow": group["dow"].iloc[0],
            "dr_high": dr_high, "dr_low": dr_low, "dr_range": dr_high - dr_low,
            "up_target_hit": up_hit, "up_pullback_count": len(up_pullbacks),
            "up_first_breakout": first_up_bo, "up_target_time": up_time,
            "dn_target_hit": dn_hit, "dn_pullback_count": len(dn_pullbacks),
            "dn_first_breakout": first_dn_bo, "dn_target_time": dn_time,
            "up_pullbacks": up_pullbacks, "dn_pullbacks": dn_pullbacks,
        })
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>ES FUTURES — ORB PROBABILITY ENGINE</h1>
    <p>Opening Range Breakout + Pullback Analysis · 100-Tick Delivery Probability · 2008–2025</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════
def build_time_options():
    times, labels = [], []
    for h in range(24):
        for m in range(0, 60, 5):
            t = time(h, m)
            times.append(t)
            labels.append(datetime(2000, 1, 1, h, m).strftime("%-I:%M %p") + " ET")
    return times, labels

time_opts, time_lbls = build_time_options()

with st.sidebar:
    st.markdown("### ⚙️ Filters")
    st.markdown("---")
    
    date_range = st.selectbox("📅 Date Range", 
        ["Entire Dataset", "Last 10 Years", "Last 5 Years",
         "Last 1 Year", "Last 6 Months", "Last 3 Months"], index=2)
    
    st.markdown("---")
    st.markdown("### 🕐 Defining Range")
    
    dr_start_idx = st.selectbox("DR Start", range(len(time_opts)),
        format_func=lambda i: time_lbls[i],
        index=time_opts.index(time(9, 0)))
    dr_start = time_opts[dr_start_idx]
    
    dr_end_idx = st.selectbox("DR End", range(len(time_opts)),
        format_func=lambda i: time_lbls[i],
        index=time_opts.index(time(9, 25)))
    dr_end = time_opts[dr_end_idx]
    
    session_end_idx = st.selectbox("Session End", range(len(time_opts)),
        format_func=lambda i: time_lbls[i],
        index=time_opts.index(time(16, 0)))
    session_end = time_opts[session_end_idx]
    
    st.markdown("---")
    st.markdown("### 📋 Day & Target")
    
    day_filter = st.selectbox("Day of Week",
        ["All Days", "Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays"])
    
    target_pts = st.slider("Target (points)", 5.0, 100.0, 25.0, 0.25, format="%.2f")
    
    st.markdown("---")
    st.markdown(f"""
    <div style='padding:12px; background:#0a0e17; border:1px solid #1a2236; border-radius:8px;
                font-family:JetBrains Mono; font-size:0.75rem; color:#94a3b8; line-height:1.6;'>
        DR: <span style='color:#3b82f6;'>{time_lbls[dr_start_idx]}</span> → 
        <span style='color:#3b82f6;'>{time_lbls[dr_end_idx]}</span><br>
        Session end: <span style='color:#3b82f6;'>{time_lbls[session_end_idx]}</span><br>
        Up target: DR High + <span style='color:#22c55e;'>{target_pts:.2f}</span><br>
        Dn target: DR Low − <span style='color:#ef4444;'>{target_pts:.2f}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ══════════════════════════════════════════════════════════════
orb = compute_orb(df, dr_start, dr_end, session_end, target_pts, day_filter, date_range)
pb = analyze_pullbacks(df, dr_start, dr_end, session_end, target_pts, day_filter, date_range)

if orb.empty:
    st.warning("No trading days found. Try widening your filters.")
    st.stop()


# ══════════════════════════════════════════════════════════════
#  STAT CARDS
# ══════════════════════════════════════════════════════════════
total = len(orb)
up_hits = orb["upside_hit"].sum()
dn_hits = orb["downside_hit"].sum()
either = orb["either_hit"].sum()
both = orb["both_hit"].sum()
up_pct = up_hits / total * 100
dn_pct = dn_hits / total * 100
either_pct = either / total * 100
both_pct = both / total * 100
avg_dr = orb["dr_range"].mean()
avg_up_min = orb.loc[orb["upside_hit"], "up_minutes"].mean()
avg_dn_min = orb.loc[orb["downside_hit"], "dn_minutes"].mean()
up_t_str = f"{avg_up_min:.0f}m" if pd.notna(avg_up_min) else "—"
dn_t_str = f"{avg_dn_min:.0f}m" if pd.notna(avg_dn_min) else "—"

k1, k2, k3, k4, k5, k6 = st.columns(6)
for col, label, value, color, sub in [
    (k1, "Trading Days", f"{total:,}", "white", f"{day_filter}"),
    (k2, "Avg DR Range", f"{avg_dr:.2f}", "blue", f"{avg_dr/0.25:.0f} ticks"),
    (k3, "▲ Upside", f"{up_pct:.1f}%", "green", f"{up_hits:,} hits · {up_t_str}"),
    (k4, "▼ Downside", f"{dn_pct:.1f}%", "red", f"{dn_hits:,} hits · {dn_t_str}"),
    (k5, "↕ Either", f"{either_pct:.1f}%", "amber", f"{either:,} of {total:,}"),
    (k6, "⇅ Both", f"{both_pct:.1f}%", "purple", f"{both:,} of {total:,}"),
]:
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value {color}">{value}</div>
            <div class="stat-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PULLBACK SUMMARY CARDS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

up_hit_days = pb[pb["up_target_hit"]]
dn_hit_days = pb[pb["dn_target_hit"]]
up_miss = pb[~pb["up_target_hit"]]
dn_miss = pb[~pb["dn_target_hit"]]

avg_up_pb_w = up_hit_days["up_pullback_count"].mean() if len(up_hit_days) > 0 else 0
avg_dn_pb_w = dn_hit_days["dn_pullback_count"].mean() if len(dn_hit_days) > 0 else 0
avg_up_pb_l = up_miss["up_pullback_count"].mean() if len(up_miss) > 0 else 0
avg_dn_pb_l = dn_miss["dn_pullback_count"].mean() if len(dn_miss) > 0 else 0

# Early/late first pullback rates
def calc_early_late(pb_df, side):
    pb_col = "up_pullbacks" if side == "up" else "dn_pullbacks"
    hit_col = "up_target_hit" if side == "up" else "dn_target_hit"
    events = []
    for _, row in pb_df.iterrows():
        pbs = row[pb_col]
        if len(pbs) > 0:
            dur = (pbs[0]["pullback_time"] - pbs[0]["breakout_time"]).total_seconds() / 60
            events.append({"early": dur <= 30, "hit": row[hit_col]})
    if not events:
        return "N/A", "N/A"
    edf = pd.DataFrame(events)
    e = edf[edf["early"]]
    l = edf[~edf["early"]]
    er = f"{e['hit'].mean()*100:.1f}%" if len(e) > 0 else "N/A"
    lr = f"{l['hit'].mean()*100:.1f}%" if len(l) > 0 else "N/A"
    return er, lr

up_early, up_late = calc_early_late(pb, "up")
dn_early, dn_late = calc_early_late(pb, "dn")

p1, p2 = st.columns(2)
with p1:
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">▲ Upside Pullback Profile</div>
        <div style='font-family:JetBrains Mono; font-size:0.85rem; color:#c8d3e0; line-height:2;'>
            Avg pullbacks (winners): <b style='color:#22c55e;'>{avg_up_pb_w:.1f}</b> &nbsp;|&nbsp;
            (losers): <b style='color:#64748b;'>{avg_up_pb_l:.1f}</b><br>
            Early PB hit rate (≤30min): <b style='color:#22c55e;'>{up_early}</b><br>
            Late PB hit rate (>30min): <b style='color:#64748b;'>{up_late}</b>
        </div>
    </div>""", unsafe_allow_html=True)

with p2:
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">▼ Downside Pullback Profile</div>
        <div style='font-family:JetBrains Mono; font-size:0.85rem; color:#c8d3e0; line-height:2;'>
            Avg pullbacks (winners): <b style='color:#ef4444;'>{avg_dn_pb_w:.1f}</b> &nbsp;|&nbsp;
            (losers): <b style='color:#64748b;'>{avg_dn_pb_l:.1f}</b><br>
            Early PB hit rate (≤30min): <b style='color:#ef4444;'>{dn_early}</b><br>
            Late PB hit rate (>30min): <b style='color:#64748b;'>{dn_late}</b>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CHARTS ROW 1: Day of Week + Pullback Distribution
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
ch1, ch2 = st.columns(2)

# ── Day of Week ──
with ch1:
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    dd = []
    for d in range(5):
        sub = orb[orb["dow"] == d]
        n = len(sub)
        if n > 0:
            dd.append({"day": dow_names[d], "up": sub["upside_hit"].mean()*100,
                       "dn": sub["downside_hit"].mean()*100, "n": n})
    if dd:
        ddf = pd.DataFrame(dd)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name="▲ Upside", x=ddf["day"], y=ddf["up"],
                              marker_color="#22c55e",
                              text=[f"{v:.1f}%" for v in ddf["up"]], textposition="outside",
                              textfont=dict(size=11, family="JetBrains Mono")))
        fig1.add_trace(go.Bar(name="▼ Downside", x=ddf["day"], y=ddf["dn"],
                              marker_color="#ef4444",
                              text=[f"{v:.1f}%" for v in ddf["dn"]], textposition="outside",
                              textfont=dict(size=11, family="JetBrains Mono")))
        fig1.update_layout(barmode="group", template="plotly_dark", height=370,
            plot_bgcolor="#0a0e17", paper_bgcolor="#0f1623",
            title=dict(text="Hit Rate by Day of Week", font=dict(size=13, family="JetBrains Mono")),
            yaxis=dict(title="Hit Rate %", range=[0,100], gridcolor="#1a2236"),
            font=dict(family="JetBrains Mono", size=11, color="#94a3b8"),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            margin=dict(l=40, r=20, t=50, b=30))
        st.plotly_chart(fig1, use_container_width=True)

# ── Pullback Distribution ──
with ch2:
    fig2 = make_subplots(rows=1, cols=2,
        subplot_titles=["▲ Upside Pullbacks", "▼ Downside Pullbacks"])
    
    for col, (hit_col, pb_col, clr) in enumerate([
        ("up_target_hit", "up_pullback_count", "#22c55e"),
        ("dn_target_hit", "dn_pullback_count", "#ef4444")], 1):
        w = pb[pb[hit_col]]
        lo = pb[~pb[hit_col]]
        labels = ["0","1","2","3","4","5","6+"]
        wc = [(w[pb_col]==i).sum() for i in range(6)] + [(w[pb_col]>=6).sum()]
        lc = [(lo[pb_col]==i).sum() for i in range(6)] + [(lo[pb_col]>=6).sum()]
        wt, lt = max(len(w),1), max(len(lo),1)
        fig2.add_trace(go.Bar(name="Winners" if col==1 else "", x=labels,
            y=[c/wt*100 for c in wc], marker_color=clr, opacity=0.9,
            showlegend=(col==1), legendgroup="w"), row=1, col=col)
        fig2.add_trace(go.Bar(name="Losers" if col==1 else "", x=labels,
            y=[c/lt*100 for c in lc], marker_color="#475569", opacity=0.6,
            showlegend=(col==1), legendgroup="l"), row=1, col=col)
    
    fig2.update_layout(barmode="group", template="plotly_dark", height=370,
        plot_bgcolor="#0a0e17", paper_bgcolor="#0f1623",
        font=dict(family="JetBrains Mono", size=11, color="#94a3b8"),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=50, b=30))
    fig2.update_yaxes(title_text="% of days", range=[0,50], gridcolor="#1a2236")
    fig2.update_xaxes(title_text="Pullback Count")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  CHARTS ROW 2: Breakout Time + Rolling Hit Rate
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
ch3, ch4 = st.columns(2)

# ── Breakout Time vs Success ──
with ch3:
    time_bins = [
        ("9:23-9:30",9*60+23,9*60+30), ("9:30-10:00",9*60+30,10*60),
        ("10:00-10:30",10*60,10*60+30), ("10:30-11:00",10*60+30,11*60),
        ("11:00-12:00",11*60,12*60), ("12:00-13:00",12*60,13*60),
        ("13:00-14:00",13*60,14*60), ("14:00-16:00",14*60,16*60)]
    
    all_up, all_dn = [], []
    for _, row in pb.iterrows():
        for p in row["up_pullbacks"]:
            bt = p["breakout_time"]
            all_up.append({"bmin": bt.hour*60+bt.minute, "hit": row["up_target_hit"]})
        for p in row["dn_pullbacks"]:
            bt = p["breakout_time"]
            all_dn.append({"bmin": bt.hour*60+bt.minute, "hit": row["dn_target_hit"]})
    
    up_t = pd.DataFrame(all_up) if all_up else pd.DataFrame(columns=["bmin","hit"])
    dn_t = pd.DataFrame(all_dn) if all_dn else pd.DataFrame(columns=["bmin","hit"])
    
    lbls, u_rates, d_rates = [], [], []
    for lbl, s, e in time_bins:
        lbls.append(lbl)
        su = up_t[(up_t["bmin"]>=s)&(up_t["bmin"]<e)]
        sd = dn_t[(dn_t["bmin"]>=s)&(dn_t["bmin"]<e)]
        u_rates.append(su["hit"].mean()*100 if len(su)>10 else None)
        d_rates.append(sd["hit"].mean()*100 if len(sd)>10 else None)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=lbls, y=u_rates, name="▲ Upside",
        line=dict(color="#22c55e", width=3), mode="lines+markers",
        marker=dict(size=8)))
    fig3.add_trace(go.Scatter(x=lbls, y=d_rates, name="▼ Downside",
        line=dict(color="#ef4444", width=3), mode="lines+markers",
        marker=dict(size=8)))
    fig3.update_layout(template="plotly_dark", height=370,
        plot_bgcolor="#0a0e17", paper_bgcolor="#0f1623",
        title=dict(text="Hit Rate by Breakout Time", font=dict(size=13, family="JetBrains Mono")),
        yaxis=dict(title="Hit Rate %", range=[0,60], gridcolor="#1a2236"),
        font=dict(family="JetBrains Mono", size=11, color="#94a3b8"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=50, b=30))
    st.plotly_chart(fig3, use_container_width=True)

# ── Rolling Hit Rate ──
with ch4:
    roll = orb.sort_values("trade_date").copy()
    roll["up_roll"] = roll["upside_hit"].rolling(60, min_periods=20).mean() * 100
    roll["dn_roll"] = roll["downside_hit"].rolling(60, min_periods=20).mean() * 100
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=roll["trade_date"], y=roll["up_roll"], name="▲ Upside",
        line=dict(color="#22c55e", width=2), fill="tozeroy", fillcolor="rgba(34,197,94,0.06)"))
    fig4.add_trace(go.Scatter(x=roll["trade_date"], y=roll["dn_roll"], name="▼ Downside",
        line=dict(color="#ef4444", width=2), fill="tozeroy", fillcolor="rgba(239,68,68,0.06)"))
    fig4.update_layout(template="plotly_dark", height=370,
        plot_bgcolor="#0a0e17", paper_bgcolor="#0f1623",
        title=dict(text="Rolling 60-Day Hit Rate", font=dict(size=13, family="JetBrains Mono")),
        yaxis=dict(title="Hit Rate %", range=[0,100], gridcolor="#1a2236"),
        xaxis=dict(gridcolor="#1a2236"),
        font=dict(family="JetBrains Mono", size=11, color="#94a3b8"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=50, b=30))
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  RAW DATA TABLE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

with st.expander("📋 View Raw Results Table"):
    disp = orb.copy()
    disp["trade_date"] = disp["trade_date"].dt.strftime("%Y-%m-%d")
    disp["dow_name"] = disp["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"})
    disp["upside_hit"] = disp["upside_hit"].map({True:"✅",False:"❌"})
    disp["downside_hit"] = disp["downside_hit"].map({True:"✅",False:"❌"})
    disp["up_minutes"] = disp["up_minutes"].apply(lambda x: f"{x:.0f}m" if pd.notna(x) else "—")
    disp["dn_minutes"] = disp["dn_minutes"].apply(lambda x: f"{x:.0f}m" if pd.notna(x) else "—")
    st.dataframe(disp[["trade_date","dow_name","dr_high","dr_low","dr_range",
                        "upside_hit","downside_hit","up_minutes","dn_minutes"]].rename(columns={
        "trade_date":"Date","dow_name":"Day","dr_high":"DR High","dr_low":"DR Low",
        "dr_range":"DR Range","upside_hit":"▲ Up","downside_hit":"▼ Dn",
        "up_minutes":"▲ Time","dn_minutes":"▼ Time"}),
        use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:20px; margin-top:12px;
            font-family:JetBrains Mono; font-size:0.65rem; color:#334155;
            border-top:1px solid #1a2236;'>
    ES Futures ORB Probability Engine · Not Financial Advice · Data: 2008–2025
</div>
""", unsafe_allow_html=True)
