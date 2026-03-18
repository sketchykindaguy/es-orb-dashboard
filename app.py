"""
ES Futures — ORB Probability Engine
Opening Range Breakout + Pullback Analysis Dashboard
Lightweight version using pre-computed daily data
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
#  DATA LOADING — lightweight precomputed file
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading pre-computed data...")
def load_data():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(app_dir, "data", "ES_precomputed.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(app_dir, "ES_precomputed.parquet")
    
    pf = pd.read_parquet(parquet_path)
    pf["trade_date"] = pd.to_datetime(pf["trade_date"])
    return pf

precomp = load_data()


# ══════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE — works on pre-computed daily data
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Computing ORB probabilities...")
def compute_orb_from_precomp(_pf, dr_start, dr_end, session_end, target_pts, day_filter, date_range):
    """
    Reconstruct DR high/low and session high/low from the pre-computed columns.
    Each column like h_0900 = high at 09:00, l_0930 = low at 09:30, c_0925 = close at 09:25
    Session highs/lows are in columns like sh_0925_1600_H and sh_0925_1600_L
    """
    pf = _pf.copy()
    
    # Date range filter
    max_date = pf["trade_date"].max()
    range_map = {
        "Entire Dataset": None, "Last 10 Years": pd.DateOffset(years=10),
        "Last 5 Years": pd.DateOffset(years=5), "Last 1 Year": pd.DateOffset(years=1),
        "Last 6 Months": pd.DateOffset(months=6), "Last 3 Months": pd.DateOffset(months=3),
    }
    offset = range_map.get(date_range)
    if offset:
        pf = pf[pf["trade_date"] >= (max_date - offset)]
    
    # Day filter
    day_map = {"Mondays": 0, "Tuesdays": 1, "Wednesdays": 2, "Thursdays": 3, "Fridays": 4}
    if day_filter != "All Days":
        pf = pf[pf["dow"] == day_map[day_filter]]
    
    if pf.empty:
        return pd.DataFrame()
    
    # Build list of 5-min time slots in the DR window
    dr_slots = []
    t = dr_start
    while t < dr_end:
        dr_slots.append(f"{t.hour:02d}{t.minute:02d}")
        # Advance by 5 minutes
        mins = t.hour * 60 + t.minute + 5
        t = time(mins // 60, mins % 60)
    
    # Build list of 5-min slots in the session window (dr_end to session_end)
    session_slots = []
    t = dr_end
    while t <= session_end:
        session_slots.append(f"{t.hour:02d}{t.minute:02d}")
        mins = t.hour * 60 + t.minute + 5
        if mins >= 24 * 60:
            break
        t = time(mins // 60, mins % 60)
    
    # Compute DR high/low from the 5-min slot columns
    dr_high_cols = [f"h_{s}" for s in dr_slots if f"h_{s}" in pf.columns]
    dr_low_cols = [f"l_{s}" for s in dr_slots if f"l_{s}" in pf.columns]
    
    if not dr_high_cols or not dr_low_cols:
        return pd.DataFrame()
    
    pf["dr_high"] = pf[dr_high_cols].max(axis=1)
    pf["dr_low"] = pf[dr_low_cols].min(axis=1)
    pf["dr_range"] = pf["dr_high"] - pf["dr_low"]
    
    # Check for session high/low from precomputed columns
    # Find closest matching precomputed session window
    dr_end_key = f"{dr_end.hour:02d}{dr_end.minute:02d}"
    se_key = f"{session_end.hour:02d}{session_end.minute:02d}"
    sh_h_col = f"sh_{dr_end_key}_{se_key}_H"
    sh_l_col = f"sh_{dr_end_key}_{se_key}_L"
    
    if sh_h_col in pf.columns and sh_l_col in pf.columns:
        pf["session_high"] = pf[sh_h_col]
        pf["session_low"] = pf[sh_l_col]
    else:
        # Fallback: compute from individual slot columns
        sess_high_cols = [f"h_{s}" for s in session_slots if f"h_{s}" in pf.columns]
        sess_low_cols = [f"l_{s}" for s in session_slots if f"l_{s}" in pf.columns]
        if not sess_high_cols or not sess_low_cols:
            return pd.DataFrame()
        pf["session_high"] = pf[sess_high_cols].max(axis=1)
        pf["session_low"] = pf[sess_low_cols].min(axis=1)
    
    # Drop rows with missing data
    pf = pf.dropna(subset=["dr_high", "dr_low", "session_high", "session_low"])
    
    if pf.empty:
        return pd.DataFrame()
    
    # Compute targets
    pf["up_target"] = pf["dr_high"] + target_pts
    pf["dn_target"] = pf["dr_low"] - target_pts
    
    pf["upside_hit"] = pf["session_high"] >= pf["up_target"]
    pf["downside_hit"] = pf["session_low"] <= pf["dn_target"]
    pf["either_hit"] = pf["upside_hit"] | pf["downside_hit"]
    pf["both_hit"] = pf["upside_hit"] & pf["downside_hit"]
    
    result = pf[["trade_date", "dow", "dr_high", "dr_low", "dr_range",
                  "upside_hit", "downside_hit", "either_hit", "both_hit"]].copy()
    
    return result


@st.cache_data(show_spinner="Analyzing pullback patterns...")
def compute_pullbacks_from_precomp(_pf, dr_start, dr_end, session_end, target_pts, day_filter, date_range):
    """
    Compute pullback counts from precomputed close prices.
    Walk through session bars using close prices to detect breakout/pullback cycles.
    """
    pf = _pf.copy()
    
    # Date range filter
    max_date = pf["trade_date"].max()
    range_map = {
        "Entire Dataset": None, "Last 10 Years": pd.DateOffset(years=10),
        "Last 5 Years": pd.DateOffset(years=5), "Last 1 Year": pd.DateOffset(years=1),
        "Last 6 Months": pd.DateOffset(months=6), "Last 3 Months": pd.DateOffset(months=3),
    }
    offset = range_map.get(date_range)
    if offset:
        pf = pf[pf["trade_date"] >= (max_date - offset)]
    
    day_map = {"Mondays": 0, "Tuesdays": 1, "Wednesdays": 2, "Thursdays": 3, "Fridays": 4}
    if day_filter != "All Days":
        pf = pf[pf["dow"] == day_map[day_filter]]
    
    if pf.empty:
        return pd.DataFrame()
    
    # DR time slots
    dr_slots = []
    t = dr_start
    while t < dr_end:
        dr_slots.append(f"{t.hour:02d}{t.minute:02d}")
        mins = t.hour * 60 + t.minute + 5
        t = time(mins // 60, mins % 60)
    
    # Session time slots
    session_slots = []
    t = dr_end
    while t <= session_end:
        session_slots.append(f"{t.hour:02d}{t.minute:02d}")
        mins = t.hour * 60 + t.minute + 5
        if mins >= 24 * 60:
            break
        t = time(mins // 60, mins % 60)
    
    dr_high_cols = [f"h_{s}" for s in dr_slots if f"h_{s}" in pf.columns]
    dr_low_cols = [f"l_{s}" for s in dr_slots if f"l_{s}" in pf.columns]
    
    if not dr_high_cols or not dr_low_cols:
        return pd.DataFrame()
    
    results = []
    
    for _, row in pf.iterrows():
        # Get DR high/low
        dr_highs = [row.get(c) for c in dr_high_cols if pd.notna(row.get(c))]
        dr_lows = [row.get(c) for c in dr_low_cols if pd.notna(row.get(c))]
        
        if not dr_highs or not dr_lows:
            continue
        
        dr_high = max(dr_highs)
        dr_low = min(dr_lows)
        dr_range = dr_high - dr_low
        
        up_target = dr_high + target_pts
        dn_target = dr_low - target_pts
        
        # Walk through session bars
        up_state, dn_state = "inside", "inside"
        up_hit, dn_hit = False, False
        up_pullbacks, dn_pullbacks = 0, 0
        first_up_bo_slot, first_dn_bo_slot = None, None
        up_bo_slot = None
        dn_bo_slot = None
        
        up_pb_events = []
        dn_pb_events = []
        
        for slot in session_slots:
            h_col = f"h_{slot}"
            l_col = f"l_{slot}"
            c_col = f"c_{slot}"
            
            h_val = row.get(h_col)
            l_val = row.get(l_col)
            c_val = row.get(c_col)
            
            if pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val):
                continue
            
            # Upside tracking
            if not up_hit:
                if h_val >= up_target:
                    up_hit = True
                if up_state == "inside" and c_val > dr_high:
                    up_state = "above"
                    up_bo_slot = slot
                    if first_up_bo_slot is None:
                        first_up_bo_slot = slot
                elif up_state == "above" and c_val <= dr_high and c_val >= dr_low:
                    up_state = "inside"
                    up_pullbacks += 1
                    up_pb_events.append({"bo_slot": up_bo_slot, "pb_slot": slot})
            
            # Downside tracking
            if not dn_hit:
                if l_val <= dn_target:
                    dn_hit = True
                if dn_state == "inside" and c_val < dr_low:
                    dn_state = "below"
                    dn_bo_slot = slot
                    if first_dn_bo_slot is None:
                        first_dn_bo_slot = slot
                elif dn_state == "below" and c_val >= dr_low and c_val <= dr_high:
                    dn_state = "inside"
                    dn_pullbacks += 1
                    dn_pb_events.append({"bo_slot": dn_bo_slot, "pb_slot": slot})
        
        # Compute first pullback timing (early vs late)
        up_first_pb_early = None
        if up_pb_events:
            bo = up_pb_events[0]["bo_slot"]
            pb = up_pb_events[0]["pb_slot"]
            bo_min = int(bo[:2]) * 60 + int(bo[2:])
            pb_min = int(pb[:2]) * 60 + int(pb[2:])
            up_first_pb_early = (pb_min - bo_min) <= 30
        
        dn_first_pb_early = None
        if dn_pb_events:
            bo = dn_pb_events[0]["bo_slot"]
            pb = dn_pb_events[0]["pb_slot"]
            bo_min = int(bo[:2]) * 60 + int(bo[2:])
            pb_min = int(pb[:2]) * 60 + int(pb[2:])
            dn_first_pb_early = (pb_min - bo_min) <= 30
        
        # First breakout time (minutes from midnight, for time-of-day analysis)
        up_bo_minute = None
        if first_up_bo_slot:
            up_bo_minute = int(first_up_bo_slot[:2]) * 60 + int(first_up_bo_slot[2:])
        dn_bo_minute = None
        if first_dn_bo_slot:
            dn_bo_minute = int(first_dn_bo_slot[:2]) * 60 + int(first_dn_bo_slot[2:])
        
        results.append({
            "trade_date": row["trade_date"],
            "dow": row["dow"],
            "dr_high": dr_high, "dr_low": dr_low, "dr_range": dr_range,
            "up_target_hit": up_hit, "up_pullback_count": up_pullbacks,
            "up_first_pb_early": up_first_pb_early, "up_bo_minute": up_bo_minute,
            "up_had_breakout": first_up_bo_slot is not None,
            "dn_target_hit": dn_hit, "dn_pullback_count": dn_pullbacks,
            "dn_first_pb_early": dn_first_pb_early, "dn_bo_minute": dn_bo_minute,
            "dn_had_breakout": first_dn_bo_slot is not None,
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
    for h in range(8, 17):
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

# Validate DR
if dr_end <= dr_start:
    st.error("DR End must be after DR Start.")
    st.stop()
if session_end <= dr_end:
    st.error("Session End must be after DR End.")
    st.stop()


# ══════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ══════════════════════════════════════════════════════════════
orb = compute_orb_from_precomp(precomp, dr_start, dr_end, session_end, target_pts, day_filter, date_range)
pb = compute_pullbacks_from_precomp(precomp, dr_start, dr_end, session_end, target_pts, day_filter, date_range)

if orb.empty:
    st.warning("No trading days found for these filters. Try widening the date range or changing the DR times.")
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

k1, k2, k3, k4, k5, k6 = st.columns(6)
for col, label, value, color, sub in [
    (k1, "Trading Days", f"{total:,}", "white", f"{day_filter}"),
    (k2, "Avg DR Range", f"{avg_dr:.2f}", "blue", f"{avg_dr/0.25:.0f} ticks"),
    (k3, "▲ Upside", f"{up_pct:.1f}%", "green", f"{up_hits:,} hits"),
    (k4, "▼ Downside", f"{dn_pct:.1f}%", "red", f"{dn_hits:,} hits"),
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

if not pb.empty:
    up_hit_days = pb[pb["up_target_hit"]]
    dn_hit_days = pb[pb["dn_target_hit"]]
    up_miss = pb[~pb["up_target_hit"]]
    dn_miss = pb[~pb["dn_target_hit"]]
    
    avg_up_pb_w = up_hit_days["up_pullback_count"].mean() if len(up_hit_days) > 0 else 0
    avg_dn_pb_w = dn_hit_days["dn_pullback_count"].mean() if len(dn_hit_days) > 0 else 0
    avg_up_pb_l = up_miss["up_pullback_count"].mean() if len(up_miss) > 0 else 0
    avg_dn_pb_l = dn_miss["dn_pullback_count"].mean() if len(dn_miss) > 0 else 0
    
    # Early/late rates
    up_early_sub = pb[(pb["up_first_pb_early"] == True)]
    up_late_sub = pb[(pb["up_first_pb_early"] == False)]
    dn_early_sub = pb[(pb["dn_first_pb_early"] == True)]
    dn_late_sub = pb[(pb["dn_first_pb_early"] == False)]
    
    up_early = f"{up_early_sub['up_target_hit'].mean()*100:.1f}%" if len(up_early_sub) > 0 else "N/A"
    up_late = f"{up_late_sub['up_target_hit'].mean()*100:.1f}%" if len(up_late_sub) > 0 else "N/A"
    dn_early = f"{dn_early_sub['dn_target_hit'].mean()*100:.1f}%" if len(dn_early_sub) > 0 else "N/A"
    dn_late = f"{dn_late_sub['dn_target_hit'].mean()*100:.1f}%" if len(dn_late_sub) > 0 else "N/A"
    
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
        st.plotly_chart(fig1, width="stretch")

with ch2:
    if not pb.empty:
        fig2 = make_subplots(rows=1, cols=2,
            subplot_titles=["▲ Upside Pullbacks", "▼ Downside Pullbacks"])
        
        for col_idx, (hit_col, pb_col, clr) in enumerate([
            ("up_target_hit", "up_pullback_count", "#22c55e"),
            ("dn_target_hit", "dn_pullback_count", "#ef4444")], 1):
            w = pb[pb[hit_col]]
            lo = pb[~pb[hit_col]]
            labels = ["0","1","2","3","4","5","6+"]
            wc = [(w[pb_col]==i).sum() for i in range(6)] + [(w[pb_col]>=6).sum()]
            lc = [(lo[pb_col]==i).sum() for i in range(6)] + [(lo[pb_col]>=6).sum()]
            wt, lt = max(len(w),1), max(len(lo),1)
            fig2.add_trace(go.Bar(name="Winners" if col_idx==1 else "", x=labels,
                y=[c/wt*100 for c in wc], marker_color=clr, opacity=0.9,
                showlegend=(col_idx==1), legendgroup="w"), row=1, col=col_idx)
            fig2.add_trace(go.Bar(name="Losers" if col_idx==1 else "", x=labels,
                y=[c/lt*100 for c in lc], marker_color="#475569", opacity=0.6,
                showlegend=(col_idx==1), legendgroup="l"), row=1, col=col_idx)
        
        fig2.update_layout(barmode="group", template="plotly_dark", height=370,
            plot_bgcolor="#0a0e17", paper_bgcolor="#0f1623",
            font=dict(family="JetBrains Mono", size=11, color="#94a3b8"),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=40, r=20, t=50, b=30))
        fig2.update_yaxes(title_text="% of days", range=[0,50], gridcolor="#1a2236")
        fig2.update_xaxes(title_text="Pullback Count")
        st.plotly_chart(fig2, width="stretch")


# ══════════════════════════════════════════════════════════════
#  CHARTS ROW 2: Breakout Time + Rolling Hit Rate
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
ch3, ch4 = st.columns(2)

with ch3:
    if not pb.empty:
        time_bins = [
            ("9:23-9:30", 9*60+23, 9*60+30), ("9:30-10:00", 9*60+30, 10*60),
            ("10:00-10:30", 10*60, 10*60+30), ("10:30-11:00", 10*60+30, 11*60),
            ("11:00-12:00", 11*60, 12*60), ("12:00-13:00", 12*60, 13*60),
            ("13:00-14:00", 13*60, 14*60), ("14:00-16:00", 14*60, 16*60)]
        
        lbls, u_rates, d_rates = [], [], []
        for lbl, s, e in time_bins:
            lbls.append(lbl)
            up_sub = pb[(pb["up_bo_minute"] >= s) & (pb["up_bo_minute"] < e)]
            dn_sub = pb[(pb["dn_bo_minute"] >= s) & (pb["dn_bo_minute"] < e)]
            u_rates.append(up_sub["up_target_hit"].mean()*100 if len(up_sub) > 10 else None)
            d_rates.append(dn_sub["dn_target_hit"].mean()*100 if len(dn_sub) > 10 else None)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=lbls, y=u_rates, name="▲ Upside",
            line=dict(color="#22c55e", width=3), mode="lines+markers", marker=dict(size=8)))
        fig3.add_trace(go.Scatter(x=lbls, y=d_rates, name="▼ Downside",
            line=dict(color="#ef4444", width=3), mode="lines+markers", marker=dict(size=8)))
        fig3.update_layout(template="plotly_dark", height=370,
            plot_bgcolor="#0a0e17", paper_bgcolor="#0f1623",
            title=dict(text="Hit Rate by Breakout Time", font=dict(size=13, family="JetBrains Mono")),
            yaxis=dict(title="Hit Rate %", range=[0,60], gridcolor="#1a2236"),
            font=dict(family="JetBrains Mono", size=11, color="#94a3b8"),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            margin=dict(l=40, r=20, t=50, b=30))
        st.plotly_chart(fig3, width="stretch")

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
    st.plotly_chart(fig4, width="stretch")


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
    st.dataframe(disp[["trade_date","dow_name","dr_high","dr_low","dr_range",
                        "upside_hit","downside_hit"]].rename(columns={
        "trade_date":"Date","dow_name":"Day","dr_high":"DR High","dr_low":"DR Low",
        "dr_range":"DR Range","upside_hit":"▲ Up","downside_hit":"▼ Dn"}),
        width="stretch", height=400)


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
