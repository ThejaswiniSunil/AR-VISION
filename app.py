import os
import time
import cv2
import gdown
import torch
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import requests
import math
from datetime import datetime
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEHAZE_MODEL_PATH = "remove_hazy_model_256x256.pth"
DEHAZE_GDRIVE_ID  = "1ji3x-KO19X2yGpT7oaUIpJ5DiCgQg8xS"
YOLO_MODEL_NAME   = "yolov8n.pt"
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "b418c8e85a223c25761a2ab362221033")

DRIVING_CLASSES = {
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "traffic light", "stop sign",
}

st.set_page_config(
    page_title="NEXTGEN VISION AI",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (unchanged from original)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background: #06090f !important;
    color: #eef4ff !important;
}
.stApp { background: #06090f !important; }
header[data-testid="stHeader"]   { display: none !important; }
.stDeployButton                  { display: none !important; }
#MainMenu                        { display: none !important; }
footer                           { display: none !important; }

[data-testid="stSidebar"] {
    background: #0b1018 !important;
    border-right: 1px solid rgba(0,255,180,0.12) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.nv-logo {
    display: flex; align-items: center; gap: 12px;
    padding: 24px 20px 18px;
    border-bottom: 1px solid rgba(0,255,180,0.10);
    margin-bottom: 6px;
}
.nv-logo-n {
    width: 38px; height: 38px; background: #00ffb4;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace; font-weight: 700; font-size: 18px;
    color: #06090f; flex-shrink: 0; border-radius: 4px;
}
.nv-logo-text  { font-size: 13px; font-weight: 800; letter-spacing: 0.1em; color: #eef4ff; line-height: 1.2; }
.nv-logo-sub   { font-size: 9px; font-family: 'Space Mono', monospace; color: #4a6070; letter-spacing: 0.1em; margin-top: 2px; }
.nv-status-bar { padding: 12px 20px 8px; border-bottom: 1px solid rgba(0,255,180,0.08); margin-bottom: 4px; }
.nv-status-item { display: flex; align-items: center; gap: 8px; font-size: 11px; font-family: 'Space Mono', monospace; color: #4a6070; padding: 3px 0; }
.nv-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.nv-dot-ok   { background: #00ffb4; box-shadow: 0 0 6px #00ffb4; }
.nv-dot-warn { background: #ff6b35; box-shadow: 0 0 6px #ff6b35; }
.nv-dot-off  { background: #2a3a4a; }
.nv-section-label { font-size: 9px; font-family: 'Space Mono', monospace; letter-spacing: 0.2em; text-transform: uppercase; color: #2e4455; padding: 14px 20px 6px; }

[data-testid="stSlider"]   { padding: 0 20px !important; }
[data-testid="stSlider"] > div > div > div > div { background: #00ffb4 !important; }
[data-testid="stToggle"]   { padding: 2px 20px !important; }
[data-testid="stToggle"] label { font-size: 12px !important; font-family: 'Space Mono', monospace !important; color: #4a6070 !important; }
[data-testid="stToggle"] [data-testid="stWidgetLabel"] p { font-size: 11px !important; font-family: 'Space Mono', monospace !important; color: #4a6070 !important; }
[data-testid="stSelectbox"] { padding: 0 20px !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background: #111820 !important; border: 1px solid rgba(0,255,180,0.18) !important; border-radius: 6px !important; color: #eef4ff !important; font-family: 'Space Mono', monospace !important; font-size: 12px !important; }
[data-testid="stRadio"] { padding: 0 20px !important; }
[data-testid="stRadio"] label { font-size: 12px !important; font-family: 'Space Mono', monospace !important; color: #4a6070 !important; }
[data-testid="stRadio"] [data-testid="stWidgetLabel"] p { font-size: 9px !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; font-family: 'Space Mono', monospace !important; color: #2e4455 !important; }
[data-testid="stTextInput"] { padding: 0 20px !important; }
[data-testid="stTextInput"] input { background: #111820 !important; border: 1px solid rgba(0,255,180,0.18) !important; border-radius: 6px !important; color: #eef4ff !important; font-family: 'Space Mono', monospace !important; font-size: 12px !important; }

.block-container { max-width: 100% !important; padding: 0 !important; }

.nv-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 36px;
    border-bottom: 1px solid rgba(0,255,180,0.10);
    background: rgba(11,16,24,0.6);
    backdrop-filter: blur(8px);
    position: sticky; top: 0; z-index: 100;
}
.nv-topbar-path { font-size: 11px; font-family: 'Space Mono', monospace; letter-spacing: 0.18em; color: #2e4455; text-transform: uppercase; }
.nv-topbar-path span { color: #00ffb4; }
.nv-chip-row { display: flex; gap: 8px; align-items: center; }
.nv-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; background: rgba(0,255,180,0.05); border: 1px solid rgba(0,255,180,0.15); border-radius: 4px; font-size: 9px; font-family: 'Space Mono', monospace; color: #4a6070; letter-spacing: 0.12em; }
.nv-chip-dot { width: 5px; height: 5px; border-radius: 50%; background: #00ffb4; }

.nv-hero { padding: 40px 36px 28px; position: relative; overflow: hidden; }
.nv-hero::before { content: ''; position: absolute; top: -60px; right: -60px; width: 300px; height: 300px; background: radial-gradient(circle, rgba(0,255,180,0.06), transparent 70%); pointer-events: none; }
.nv-hero-headline { font-size: 52px; font-weight: 800; line-height: 0.95; letter-spacing: -0.04em; color: #eef4ff; margin-bottom: 14px; }
.nv-hero-headline em { font-style: normal; color: #00ffb4; display: block; }
.nv-hero-desc { font-size: 13px; font-family: 'Space Mono', monospace; color: #4a6070; line-height: 1.8; max-width: 520px; margin-bottom: 20px; }
.nv-badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.nv-badge { font-size: 9px; font-family: 'Space Mono', monospace; letter-spacing: 0.12em; padding: 5px 10px; border: 1px solid rgba(0,255,180,0.3); border-radius: 3px; color: #00ffb4; text-transform: uppercase; }

.nv-metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; padding: 0 36px 28px; }
.nv-metric { background: #0b1018; border: 1px solid rgba(0,255,180,0.12); border-radius: 8px; padding: 18px 20px; position: relative; overflow: hidden; }
.nv-metric::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #00ffb4, #00c8ff); }
.nv-metric-val { font-size: 22px; font-weight: 800; font-family: 'Space Mono', monospace; letter-spacing: -0.02em; color: #eef4ff; margin-bottom: 6px; margin-top: 4px; }
.nv-metric-lbl { font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase; color: #2e4455; font-family: 'Space Mono', monospace; }
.nv-metric-sub { font-size: 10px; color: #00ffb4; font-family: 'Space Mono', monospace; margin-top: 4px; }

.nv-content { padding: 0 36px 40px; }
.nv-section-head { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
.nv-section-title { font-size: 20px; font-weight: 800; letter-spacing: -0.02em; color: #eef4ff; }
.nv-section-tag { font-size: 9px; font-family: 'Space Mono', monospace; letter-spacing: 0.18em; padding: 4px 9px; border: 1px solid rgba(0,255,180,0.3); color: #00ffb4; border-radius: 3px; text-transform: uppercase; }

.nv-info { background: rgba(0,200,255,0.06); border: 1px solid rgba(0,200,255,0.2); border-radius: 6px; padding: 12px 16px; font-size: 12px; font-family: 'Space Mono', monospace; color: rgba(0,200,255,0.85); margin-bottom: 18px; line-height: 1.7; }
.nv-warn { background: rgba(255,107,53,0.07); border: 1px solid rgba(255,107,53,0.22); border-radius: 6px; padding: 12px 16px; font-size: 12px; font-family: 'Space Mono', monospace; color: rgba(255,160,80,0.9); margin-bottom: 18px; line-height: 1.7; }

[data-testid="stFileUploader"] { background: rgba(0,255,180,0.02) !important; border: 1px dashed rgba(0,255,180,0.28) !important; border-radius: 10px !important; padding: 1.5rem !important; }
[data-testid="stFileUploader"] label { font-family: 'Space Mono', monospace !important; color: #4a6070 !important; font-size: 12px !important; }
[data-testid="stFileUploader"] button { background: rgba(0,255,180,0.10) !important; border: 1px solid rgba(0,255,180,0.30) !important; color: #00ffb4 !important; border-radius: 5px !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; font-weight: 700 !important; }

[data-testid="stImage"] img { border-radius: 8px !important; border: 1px solid rgba(0,255,180,0.15) !important; }

.nv-img-card { background: #0b1018; border: 1px solid rgba(0,255,180,0.12); border-radius: 10px; overflow: hidden; }
.nv-img-card-head { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-bottom: 1px solid rgba(0,255,180,0.10); }
.nv-img-card-title { font-size: 9px; font-family: 'Space Mono', monospace; letter-spacing: 0.18em; text-transform: uppercase; color: #4a6070; }
.nv-img-status { font-size: 9px; font-family: 'Space Mono', monospace; color: #00ffb4; background: rgba(0,255,180,0.10); padding: 2px 8px; border-radius: 3px; letter-spacing: 0.1em; }
.nv-img-status-warn { color: #ff6b35; background: rgba(255,107,53,0.10); }

.nv-vis-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-top: 20px; }
.nv-vis-bar { background: #0b1018; border: 1px solid rgba(0,255,180,0.12); border-radius: 8px; padding: 14px 16px; }
.nv-vis-lbl { font-size: 9px; font-family: 'Space Mono', monospace; letter-spacing: 0.16em; text-transform: uppercase; color: #2e4455; margin-bottom: 10px; }
.nv-vis-track { height: 3px; background: rgba(255,255,255,0.06); border-radius: 2px; margin-bottom: 8px; overflow: hidden; }
.nv-vis-fill { height: 100%; border-radius: 2px; }
.nv-vis-val { font-size: 16px; font-weight: 700; font-family: 'Space Mono', monospace; }

.nv-det-table { background: #0b1018; border: 1px solid rgba(0,255,180,0.12); border-radius: 10px; overflow: hidden; margin-top: 20px; }
.nv-det-table-head { display: flex; align-items: center; justify-content: space-between; padding: 12px 18px; border-bottom: 1px solid rgba(0,255,180,0.10); }
.nv-det-title { font-size: 10px; font-family: 'Space Mono', monospace; letter-spacing: 0.18em; text-transform: uppercase; color: #4a6070; }
.nv-det-count { font-size: 10px; font-family: 'Space Mono', monospace; color: #00ffb4; }

.nv-weather-card {
    background: linear-gradient(135deg, #0b1018, #0d1620);
    border: 1px solid rgba(0,255,180,0.15);
    border-radius: 12px; padding: 20px;
    margin: 0 20px 16px; position: relative; overflow: hidden;
}
.nv-weather-card::before {
    content: ''; position: absolute; top: -30px; right: -30px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(0,200,255,0.08), transparent 70%);
}
.nv-weather-city { font-size: 11px; font-family: 'Space Mono', monospace; color: #4a6070; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 6px; }
.nv-weather-temp { font-size: 32px; font-weight: 800; color: #eef4ff; line-height: 1; margin-bottom: 4px; }
.nv-weather-desc { font-size: 11px; font-family: 'Space Mono', monospace; color: #00ffb4; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.1em; }
.nv-weather-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.nv-weather-stat { font-size: 10px; font-family: 'Space Mono', monospace; color: #4a6070; }
.nv-weather-stat span { color: #eef4ff; font-weight: 700; }
.nv-weather-alert { background: rgba(255,107,53,0.12); border: 1px solid rgba(255,107,53,0.3); border-radius: 6px; padding: 8px 10px; font-size: 10px; font-family: 'Space Mono', monospace; color: #ff9a6b; margin-top: 10px; }
.nv-weather-safe  { background: rgba(0,255,180,0.08); border: 1px solid rgba(0,255,180,0.2); border-radius: 6px; padding: 8px 10px; font-size: 10px; font-family: 'Space Mono', monospace; color: #00ffb4; margin-top: 10px; }

.nv-analytics-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 20px; }
.nv-analytics-card { background: #0b1018; border: 1px solid rgba(0,255,180,0.12); border-radius: 10px; padding: 20px; position: relative; overflow: hidden; }
.nv-analytics-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #00ffb4, #00c8ff); }
.nv-analytics-val { font-size: 28px; font-weight: 800; font-family: 'Space Mono', monospace; color: #eef4ff; margin-bottom: 4px; }
.nv-analytics-lbl { font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase; color: #2e4455; font-family: 'Space Mono', monospace; }
.nv-analytics-sub { font-size: 11px; color: #00ffb4; font-family: 'Space Mono', monospace; margin-top: 6px; }

.stButton > button { background: linear-gradient(135deg, rgba(0,255,180,0.15), rgba(0,200,255,0.10)) !important; border: 1px solid rgba(0,255,180,0.35) !important; color: #00ffb4 !important; border-radius: 7px !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; font-size: 12px !important; letter-spacing: 0.12em !important; padding: 10px 24px !important; transition: all 0.15s !important; }
.stButton > button:hover { background: rgba(0,255,180,0.22) !important; border-color: rgba(0,255,180,0.60) !important; }

[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #00ffb4, #00c8ff) !important; border-radius: 2px !important; }
[data-testid="stProgressBar"] > div { background: rgba(0,255,180,0.10) !important; border-radius: 2px !important; }
[data-testid="stMetric"] { background: #0b1018 !important; border: 1px solid rgba(0,255,180,0.12) !important; border-radius: 8px !important; padding: 16px 18px !important; position: relative !important; overflow: hidden !important; }
[data-testid="stMetric"]::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #00ffb4, #00c8ff); }
[data-testid="stMetricLabel"] { font-size: 9px !important; font-family: 'Space Mono', monospace !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; color: #2e4455 !important; }
[data-testid="stMetricValue"] { font-size: 20px !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; color: #eef4ff !important; }
[data-testid="stVideo"] video { border-radius: 10px !important; border: 1px solid rgba(0,255,180,0.15) !important; }
[data-testid="stAlert"] { background: #0b1018 !important; border: 1px solid rgba(0,255,180,0.18) !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-size: 12px !important; }
[data-testid="stExpander"] { background: #0b1018 !important; border: 1px solid rgba(0,255,180,0.12) !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary { font-family: 'Space Mono', monospace !important; font-size: 11px !important; color: #4a6070 !important; letter-spacing: 0.1em !important; }
[data-testid="stDownloadButton"] > button { background: rgba(0,255,180,0.10) !important; border: 1px solid rgba(0,255,180,0.30) !important; color: #00ffb4 !important; border-radius: 6px !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; font-weight: 700 !important; }
hr { border-color: rgba(0,255,180,0.10) !important; margin: 8px 0 !important; }
.stSuccess { background: rgba(0,255,180,0.06) !important; border: 1px solid rgba(0,255,180,0.2) !important; border-radius: 6px !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; }
.stError   { background: rgba(255,60,60,0.06) !important; border: 1px solid rgba(255,60,60,0.2) !important; border-radius: 6px !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# WEATHER HELPERS
# ─────────────────────────────────────────────
def get_weather(city: str, api_key: str):
    if not api_key or not city:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def weather_to_road_condition(weather_data):
    if not weather_data:
        return "Unknown", "low", "❓"
    main = weather_data.get("weather", [{}])[0].get("main", "").lower()
    vis  = weather_data.get("visibility", 10000)
    mapping = {
        "fog":          ("DENSE FOG",      "high",   "🌫️"),
        "mist":         ("MIST / LOW VIS", "medium", "🌁"),
        "haze":         ("HAZE",           "medium", "🌫️"),
        "smoke":        ("SMOKE",          "high",   "💨"),
        "rain":         ("RAIN",           "medium", "🌧️"),
        "drizzle":      ("DRIZZLE",        "low",    "🌦️"),
        "thunderstorm": ("THUNDERSTORM",   "high",   "⛈️"),
        "snow":         ("SNOW / ICE",     "high",   "❄️"),
        "sand":         ("SANDSTORM",      "high",   "🌪️"),
        "dust":         ("DUST",           "medium", "💨"),
        "clear":        ("CLEAR",          "low",    "☀️"),
        "clouds":       ("OVERCAST",       "low",    "☁️"),
    }
    for key, val in mapping.items():
        if key in main:
            return val
    if vis < 1000:
        return "VERY LOW VIS", "high", "🌫️"
    return "NORMAL", "low", "✅"

def render_weather_sidebar(weather_data, city):
    if not weather_data:
        st.markdown(f"""
        <div class="nv-weather-card">
          <div class="nv-weather-city">Weather — {city or 'No city set'}</div>
          <div style="font-size:11px;font-family:'Space Mono',monospace;color:#2e4455;margin-top:8px">
            Enter city + API key to load live weather conditions.
          </div>
        </div>""", unsafe_allow_html=True)
        return

    temp  = round(weather_data["main"]["temp"])
    feels = round(weather_data["main"]["feels_like"])
    hum   = weather_data["main"]["humidity"]
    wind  = round(weather_data["wind"]["speed"] * 3.6, 1)
    vis   = weather_data.get("visibility", 10000) // 1000
    desc  = weather_data["weather"][0]["description"].upper()
    cond, severity, icon = weather_to_road_condition(weather_data)
    alert_cls = "nv-weather-alert" if severity in ("medium","high") else "nv-weather-safe"
    alert_msg = {
        "high":   f"⚠️ HIGH RISK — {cond}. Dehazing active.",
        "medium": f"⚡ MODERATE — {cond}. Stay alert.",
        "low":    f"✅ CLEAR — {cond}. Good visibility.",
    }[severity]

    st.markdown(f"""
    <div class="nv-weather-card">
      <div class="nv-weather-city">{icon} {city.upper()}</div>
      <div class="nv-weather-temp">{temp}°C</div>
      <div class="nv-weather-desc">{desc}</div>
      <div class="nv-weather-grid">
        <div class="nv-weather-stat">Feels like <span>{feels}°C</span></div>
        <div class="nv-weather-stat">Humidity <span>{hum}%</span></div>
        <div class="nv-weather-stat">Wind <span>{wind} km/h</span></div>
        <div class="nv-weather-stat">Visibility <span>{vis} km</span></div>
      </div>
      <div class="{alert_cls}">{alert_msg}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# NAV HUD OVERLAY
# ─────────────────────────────────────────────
NAV_WAYPOINTS = [
    ("Head North on Highway 1", 1200),
    ("Turn right — Exit 14B", 800),
    ("Merge onto Ring Road", 2100),
    ("Keep left — tunnel ahead", 400),
    ("Turn left — destination 500m", 500),
    ("Arriving at destination", 0),
]

def draw_nav_hud(frame_bgr, frame_idx, total_frames, route_name="Demo Route"):
    h, w = frame_bgr.shape[:2]
    out  = frame_bgr.copy()
    progress   = frame_idx / max(total_frames, 1)
    wp_idx     = min(int(progress * len(NAV_WAYPOINTS)), len(NAV_WAYPOINTS) - 1)
    wp_text, _ = NAV_WAYPOINTS[wp_idx]
    dist_rem   = max(0, int((1 - progress) * 12400))
    box_w, box_h = 280, 90
    x1 = w - box_w - 12
    y1 = 12
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (8, 12, 20), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
    cv2.rectangle(out, (x1, y1), (x1 + box_w, y1 + box_h), (0, 255, 180), 1)
    cv2.line(out, (x1, y1), (x1 + box_w, y1), (0, 255, 180), 2)
    ax, ay = x1 + 22, y1 + 44
    arrow_pts = np.array([[ax, ay-16],[ax-12, ay+8],[ax+12, ay+8]], np.int32)
    cv2.fillPoly(out, [arrow_pts], (0, 255, 180))
    cv2.putText(out, wp_text[:28], (x1 + 42, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (238, 244, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{dist_rem // 1000:.1f} km remaining  |  {route_name}",
                (x1 + 8, y1 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (74, 96, 112), 1, cv2.LINE_AA)
    bar_x1 = x1 + 8
    bar_x2 = x1 + box_w - 8
    bar_y  = y1 + box_h - 10
    cv2.rectangle(out, (bar_x1, bar_y - 3), (bar_x2, bar_y + 3), (20, 30, 40), -1)
    fill_x = int(bar_x1 + (bar_x2 - bar_x1) * progress)
    cv2.rectangle(out, (bar_x1, bar_y - 3), (fill_x, bar_y + 3), (0, 255, 180), -1)
    return out


# ─────────────────────────────────────────────
# UI HELPER COMPONENTS
# ─────────────────────────────────────────────
def topbar(page: str):
    parts = page.split("/")
    path_html = " <span style='color:#2e4455'>/</span> ".join(
        [f"<span style='color:#00ffb4'>{p}</span>" if i == len(parts)-1
         else f"<span style='color:#2e4455'>{p}</span>"
         for i, p in enumerate(parts)]
    )
    st.markdown(f"""
    <div class="nv-topbar">
      <div class="nv-topbar-path">/ {path_html}</div>
      <div class="nv-chip-row">
        <div class="nv-chip"><div class="nv-chip-dot"></div>SYSTEM ONLINE</div>
        <div class="nv-chip">DCP + RESNET</div>
        <div class="nv-chip">YOLOV8N</div>
      </div>
    </div>""", unsafe_allow_html=True)

def hero():
    st.markdown("""
    <div class="nv-hero">
      <div class="nv-hero-headline">NEXTGEN<em>VISION AI</em></div>
      <div class="nv-hero-desc">
        Real-time AR vision enhancement for adverse weather driving conditions.<br>
        Visibility restoration · hazard detection · navigation HUD · trip analytics.
      </div>
      <div class="nv-badge-row">
        <div class="nv-badge">DCP + ResNet Dehazing</div>
        <div class="nv-badge">YOLOv8 Detection</div>
        <div class="nv-badge">Nav HUD</div>
        <div class="nv-badge">Trip Analytics</div>
        <div class="nv-badge">Live Weather</div>
      </div>
    </div>""", unsafe_allow_html=True)

def metric_cards(device, enable_detection, inference_size):
    yolo_val = "ON" if enable_detection else "OFF"
    yolo_sub = "conf ≥ 0.35" if enable_detection else "disabled"
    st.markdown(f"""
    <div class="nv-metrics">
      <div class="nv-metric">
        <div class="nv-metric-lbl">Dehazing Model</div>
        <div class="nv-metric-val">READY</div>
        <div class="nv-metric-sub">DCP + ResNet</div>
      </div>
      <div class="nv-metric">
        <div class="nv-metric-lbl">Compute Device</div>
        <div class="nv-metric-val">{str(device).upper()}</div>
        <div class="nv-metric-sub">torch — active</div>
      </div>
      <div class="nv-metric">
        <div class="nv-metric-lbl">YOLO Detection</div>
        <div class="nv-metric-val">{yolo_val}</div>
        <div class="nv-metric-sub">{yolo_sub}</div>
      </div>
      <div class="nv-metric">
        <div class="nv-metric-lbl">Inference Size</div>
        <div class="nv-metric-val">{inference_size}px</div>
        <div class="nv-metric-sub">balanced mode</div>
      </div>
    </div>""", unsafe_allow_html=True)

def section_head(title: str, tag: str):
    st.markdown(f"""
    <div class="nv-section-head">
      <div class="nv-section-title">{title}</div>
      <div class="nv-section-tag">{tag}</div>
    </div>""", unsafe_allow_html=True)

def info_box(msg: str):
    st.markdown(f'<div class="nv-info">ℹ &nbsp; {msg}</div>', unsafe_allow_html=True)

def warn_box(msg: str):
    st.markdown(f'<div class="nv-warn">⚡ &nbsp; {msg}</div>', unsafe_allow_html=True)

def vis_bars(orig_score, enh_score, proc_time, det_count):
    o_pct = min(orig_score, 100)
    e_pct = min(enh_score, 100)
    t_pct = min(int(proc_time / 5 * 100), 100)
    d_pct = min(det_count * 10, 100)
    st.markdown(f"""
    <div class="nv-vis-grid">
      <div class="nv-vis-bar">
        <div class="nv-vis-lbl">Orig. Visibility</div>
        <div class="nv-vis-track"><div class="nv-vis-fill" style="width:{o_pct}%;background:#ff6b35"></div></div>
        <div class="nv-vis-val" style="color:#ff6b35">{o_pct}%</div>
      </div>
      <div class="nv-vis-bar">
        <div class="nv-vis-lbl">Enhanced Vis.</div>
        <div class="nv-vis-track"><div class="nv-vis-fill" style="width:{e_pct}%;background:#00ffb4"></div></div>
        <div class="nv-vis-val" style="color:#00ffb4">{e_pct}%</div>
      </div>
      <div class="nv-vis-bar">
        <div class="nv-vis-lbl">Process Time</div>
        <div class="nv-vis-track"><div class="nv-vis-fill" style="width:{t_pct}%;background:#00c8ff"></div></div>
        <div class="nv-vis-val" style="color:#00c8ff">{proc_time:.2f}s</div>
      </div>
      <div class="nv-vis-bar">
        <div class="nv-vis-lbl">Objects Found</div>
        <div class="nv-vis-track"><div class="nv-vis-fill" style="width:{d_pct}%;background:#00ffb4"></div></div>
        <div class="nv-vis-val" style="color:#00ffb4">{det_count}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def sidebar_logo(dehaze_ready: bool, yolo_ready: bool):
    dehaze_dot = "nv-dot-ok" if dehaze_ready else "nv-dot-warn"
    yolo_dot   = "nv-dot-ok" if yolo_ready   else "nv-dot-off"
    dehaze_txt = "Dehazing model — ready" if dehaze_ready else "Dehazing model — missing"
    yolo_txt   = "YOLOv8 — available"    if yolo_ready   else "YOLOv8 — not installed"
    st.markdown(f"""
    <div class="nv-logo">
      <div class="nv-logo-n">N</div>
      <div>
        <div class="nv-logo-text">NEXTGEN VISION</div>
        <div class="nv-logo-sub">AR ENHANCEMENT SYSTEM v2</div>
      </div>
    </div>
    <div class="nv-status-bar">
      <div class="nv-status-item"><div class="nv-dot {dehaze_dot}"></div>{dehaze_txt}</div>
      <div class="nv-status-item"><div class="nv-dot {yolo_dot}"></div>{yolo_txt}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ANALYTICS RENDERER
# ─────────────────────────────────────────────
def render_trip_analytics(analytics: dict, weather_data=None):
    section_head("Trip Analytics", "POST-TRIP REPORT")
    frames_proc  = analytics.get("frames_processed", 0)
    total_dets   = analytics.get("total_detections", 0)
    avg_orig_vis = analytics.get("avg_orig_visibility", 0)
    avg_enh_vis  = analytics.get("avg_enh_visibility", 0)
    vis_gain     = avg_enh_vis - avg_orig_vis
    total_time   = analytics.get("total_time", 0)
    vis_history  = analytics.get("visibility_history", [])
    det_history  = analytics.get("detection_history", [])
    class_counts = analytics.get("class_counts", {})

    fog_pct = max(0, 100 - avg_orig_vis)
    fog_label = "SEVERE" if fog_pct > 60 else "MODERATE" if fog_pct > 30 else "LIGHT"
    fog_color = "#ff6b35" if fog_pct > 60 else "#ffd700" if fog_pct > 30 else "#00ffb4"

    st.markdown(f"""
    <div class="nv-analytics-grid">
      <div class="nv-analytics-card">
        <div class="nv-analytics-lbl">Visibility Gain</div>
        <div class="nv-analytics-val" style="color:#00ffb4">+{vis_gain:.0f}%</div>
        <div class="nv-analytics-sub">{avg_orig_vis:.0f}% → {avg_enh_vis:.0f}% avg</div>
      </div>
      <div class="nv-analytics-card">
        <div class="nv-analytics-lbl">Fog Severity</div>
        <div class="nv-analytics-val" style="color:{fog_color}">{fog_label}</div>
        <div class="nv-analytics-sub">{fog_pct:.0f}% obstruction avg</div>
      </div>
      <div class="nv-analytics-card">
        <div class="nv-analytics-lbl">Hazards Detected</div>
        <div class="nv-analytics-val">{total_dets}</div>
        <div class="nv-analytics-sub">across {frames_proc} frames</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if vis_history:
        chart_df = pd.DataFrame({
            "Frame":    list(range(len(vis_history))),
            "Original Visibility (%)": [v[0] for v in vis_history],
            "Enhanced Visibility (%)": [v[1] for v in vis_history],
        }).set_index("Frame")
        st.markdown('<div style="margin-bottom:8px"><span style="font-size:9px;font-family:Space Mono,monospace;letter-spacing:0.18em;color:#2e4455;text-transform:uppercase">Visibility Over Time</span></div>', unsafe_allow_html=True)
        st.line_chart(chart_df, color=["#ff6b35", "#00ffb4"])

    if det_history:
        det_df = pd.DataFrame({
            "Frame":            list(range(len(det_history))),
            "Objects Detected": det_history,
        }).set_index("Frame")
        st.markdown('<div style="margin-top:16px;margin-bottom:8px"><span style="font-size:9px;font-family:Space Mono,monospace;letter-spacing:0.18em;color:#2e4455;text-transform:uppercase">Object Detections Per Frame</span></div>', unsafe_allow_html=True)
        st.bar_chart(det_df, color="#00c8ff")

    if class_counts:
        st.markdown('<div style="margin-top:16px;margin-bottom:8px"><span style="font-size:9px;font-family:Space Mono,monospace;letter-spacing:0.18em;color:#2e4455;text-transform:uppercase">Detected Object Classes</span></div>', unsafe_allow_html=True)
        class_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"]).sort_values("Count", ascending=False)
        st.dataframe(class_df, use_container_width=True, hide_index=True)

    if fog_pct > 60 or (weather_data and weather_to_road_condition(weather_data)[1] == "high"):
        st.markdown('<div class="nv-warn" style="margin-top:16px">⚠️ &nbsp; HIGH HAZARD TRIP — Dense fog detected. Dehazing improved visibility significantly. Recommendation: reduce speed, increase following distance, use fog lights.</div>', unsafe_allow_html=True)
    elif fog_pct > 30:
        st.markdown('<div class="nv-info" style="margin-top:16px">ℹ &nbsp; MODERATE CONDITIONS — Partial fog detected during trip. Enhancement active throughout. Continue monitoring visibility.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:rgba(0,255,180,0.06);border:1px solid rgba(0,255,180,0.2);border-radius:6px;padding:12px 16px;font-size:12px;font-family:Space Mono,monospace;color:#00ffb4;margin-top:16px;line-height:1.7">✅ &nbsp; GOOD CONDITIONS — Minimal fog detected. System operating in standard mode.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
def download_dehaze_model_if_needed():
    if os.path.exists(DEHAZE_MODEL_PATH):
        return True, None
    try:
        gdown.download(f"https://drive.google.com/uc?id={DEHAZE_GDRIVE_ID}", DEHAZE_MODEL_PATH, quiet=False)
        return (os.path.exists(DEHAZE_MODEL_PATH), None if os.path.exists(DEHAZE_MODEL_PATH) else "Downloaded but file not found")
    except Exception as e:
        return False, str(e)

class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = nn.AvgPool2d(kernel_size=2*r+1, stride=1, padding=r)

    def forward(self, I, p):
        N       = self.boxfilter(torch.ones(p.size(), device=p.device, dtype=p.dtype))
        mean_I  = self.boxfilter(I) / N
        mean_p  = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip  = mean_Ip - mean_I * mean_p
        mean_II = self.boxfilter(I * I) / N
        var_I   = mean_II - mean_I * mean_I
        a       = cov_Ip / (var_I + self.eps)
        b       = mean_p - a * mean_I
        return (self.boxfilter(a) / N) * I + self.boxfilter(b) / N

class DCPDehazeGenerator(nn.Module):
    def __init__(self, win_size=15, r=40, eps=1e-3):
        super().__init__()
        self.guided_filter     = GuidedFilter(r=r, eps=eps)
        self.neighborhood_size = win_size
        self.omega             = 0.95

    def get_dark_channel(self, img, w):
        img, _ = torch.min(img, dim=1)
        img = torch.unsqueeze(img, dim=1)
        p = int(np.floor(w / 2))
        pads = [p, p-1, p, p-1] if w % 2 == 0 else [p, p, p, p]
        return -F.max_pool2d(-F.pad(img, pads, mode='replicate'), kernel_size=w, stride=1)

    def atmospheric_light(self, img, dark_img):
        num, chl, h, w = img.shape
        top = max(int(0.001 * h * w), 1)
        A = torch.zeros(num, chl, 1, 1, device=img.device, dtype=img.dtype)
        for n in range(num):
            _, idx = dark_img[n, 0].reshape(h * w).sort(descending=True)
            for c in range(chl):
                A[n, c, 0, 0] = torch.mean(img[n, c].reshape(h * w)[idx[:top]])
        return A

    def forward(self, x):
        guidance = (0.2989*x[:,0] + 0.5870*x[:,1] + 0.1140*x[:,2]) if x.shape[1] > 1 else x[:,0]
        guidance = torch.unsqueeze((guidance + 1) / 2, 1)
        img = (x + 1) / 2
        _, _, h, w = img.shape
        dark = self.get_dark_channel(img, self.neighborhood_size)
        A = self.atmospheric_light(img, dark)
        map_A = A.repeat(1, 1, h, w).clamp(min=1e-6)
        trans = (1 - self.omega * self.get_dark_channel(img / map_A, self.neighborhood_size)).clamp(0.05, 1.0)
        T = self.guided_filter(guidance, trans).clamp(0.05, 1.0)
        return ((img - map_A) / T.repeat(1, 3, 1, 1) + map_A).clamp(0, 1)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        block = []
        for i in range(2):
            if padding_type == 'reflect':
                block += [nn.ReflectionPad2d(1)]; p = 0
            elif padding_type == 'replicate':
                block += [nn.ReplicationPad2d(1)]; p = 0
            else:
                p = 1
            block += [nn.Conv2d(dim, dim, 3, 1, p, bias=use_bias), norm_layer(dim)]
            if i == 0:
                block += [nn.ReLU(True)]
                if use_dropout:
                    block += [nn.Dropout(0.5)]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, padding_type='reflect'):
        super().__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d or
                    (hasattr(norm_layer, 'func') and norm_layer.func == nn.InstanceNorm2d))
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, 7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]
        for i in range(2):
            m = 2**i
            model += [nn.Conv2d(ngf*m, ngf*m*2, 3, 2, 1, bias=use_bias),
                      norm_layer(ngf*m*2), nn.ReLU(True)]
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf*4, padding_type, norm_layer, use_dropout, use_bias)]
        for i in range(2):
            m = 2**(2-i)
            model += [nn.ConvTranspose2d(ngf*m, ngf*m//2, 3, 2, 1, output_padding=1, bias=use_bias),
                      norm_layer(ngf*m//2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7, padding=0, bias=use_bias),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return torch.clamp(self.model(x), -1, 1)


# ─────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_dehaze_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcp    = DCPDehazeGenerator().to(device).eval()
    resnet = ResnetGenerator(3, 3, norm_layer=nn.InstanceNorm2d).to(device)
    ckpt   = torch.load(DEHAZE_MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict):
        key = next((k for k in ['params', 'state_dict', 'model', 'net_g', 'generator'] if k in ckpt), None)
        sd  = ckpt[key] if key else ckpt
    else:
        sd = ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = resnet.load_state_dict(sd, strict=False)
    resnet.eval()
    return dcp, resnet, device, missing, unexpected

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    return YOLO(YOLO_MODEL_NAME) if YOLO_AVAILABLE else None


# ─────────────────────────────────────────────
# PROCESSING HELPERS
# ─────────────────────────────────────────────
def bgr_to_tensor(img, size):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) * 2 - 1

def tensor_to_bgr(tensor, hw):
    out = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    bgr = cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)

def dehaze_image(img_bgr, strength=1.0, dcp_only=False, inference_size=192):
    h, w = img_bgr.shape[:2]
    dcp, resnet, device, _, _ = load_dehaze_models()
    x = bgr_to_tensor(img_bgr, inference_size).to(device)
    with torch.no_grad():
        dcp_out = dcp(x)
        refined = dcp_out if dcp_only else (resnet(dcp_out) + 1) / 2
        result  = tensor_to_bgr(refined, (h, w))
    return cv2.addWeighted(img_bgr, 1 - strength, result, strength, 0) if strength < 1 else result

def detect_objects_yolo(img_bgr, conf_threshold=0.35, only_driving_classes=True, draw_ar_style=True):
    model = load_yolo_model()
    if model is None:
        return img_bgr, []
    results   = model(img_bgr, conf=conf_threshold, verbose=False)
    result    = results[0]
    annotated = img_bgr.copy()
    detections = []

    if result.boxes is None:
        return annotated, detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf_v = float(box.conf[0])
        name   = model.names[cls_id]
        if only_driving_classes and name not in DRIVING_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        detections.append({"class": name, "confidence": round(conf_v, 2), "box": [x1, y1, x2, y2]})

        color = (0, 255, 150)
        if name in ["person", "motorcycle", "bicycle"]:
            color = (0, 80, 255)
        elif name in ["traffic light", "stop sign"]:
            color = (0, 212, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label   = f"{name.upper()} {conf_v:.2f}"
        label_y = max(y1 - 10, 25)
        cv2.rectangle(annotated, (x1, label_y-22), (x1 + max(120, len(label)*11), label_y+4), color, -1)
        cv2.putText(annotated, label, (x1+5, label_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (5, 10, 18), 2, cv2.LINE_AA)
        cv2.circle(annotated, ((x1+x2)//2, (y1+y2)//2), 4, color, -1)

    return annotated, detections

def draw_system_overlay(img_bgr, mode='IMAGE', fps=None, inference_time=None, detection_count=0):
    out = img_bgr.copy()
    cv2.putText(out, f'NEXTGEN VISION AI | {mode}', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 180), 2, cv2.LINE_AA)
    line = f'Objects: {detection_count}'
    line += f' | FPS: {fps:.1f}' if fps is not None else ''
    line += f' | Time: {inference_time:.2f}s' if inference_time is not None else ''
    cv2.putText(out, line, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
    return out

def visibility_score(img_bgr):
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    sharp    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(100, max(0, int(contrast * 1.4 + (sharp ** 0.5) * 2))), round(contrast, 2), round(sharp, 2)

def make_split_frame(orig_bgr, dehazed_bgr):
    h1, w1 = orig_bgr.shape[:2]
    h2, w2 = dehazed_bgr.shape[:2]
    if h1 != h2 or w1 != w2:
        dehazed_bgr = cv2.resize(dehazed_bgr, (w1, h1))
    divider = np.zeros((h1, 4, 3), dtype=np.uint8)
    divider[:] = (0, 255, 180)
    combined = np.hstack([orig_bgr, divider, dehazed_bgr])
    cv2.rectangle(combined, (0, 0), (160, 28), (8, 12, 20), -1)
    cv2.putText(combined, "ORIGINAL", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 107, 53), 2, cv2.LINE_AA)
    dx = w1 + 4
    cv2.rectangle(combined, (dx, 0), (dx + 160, 28), (8, 12, 20), -1)
    cv2.putText(combined, "DEHAZED", (dx + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2, cv2.LINE_AA)
    return combined

def process_pipeline(frame, strength, dcp_only, inference_size, enable_detection,
                     conf, only_classes, ar_style, mode,
                     frame_idx=0, total_frames=1, show_nav=True, route_name="Demo Route"):
    t0 = time.time()
    dehazed = dehaze_image(frame, strength, dcp_only, inference_size)
    dt = time.time() - t0
    t1 = time.time()
    final, dets = detect_objects_yolo(dehazed, conf, only_classes, ar_style) if enable_detection else (dehazed.copy(), [])
    yt = time.time() - t1
    if show_nav:
        final = draw_nav_hud(final, frame_idx, total_frames, route_name)
    final = draw_system_overlay(final, mode=mode, inference_time=dt+yt, detection_count=len(dets))
    return dehazed, final, dets, dt, yt


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    sidebar_logo(
        dehaze_ready=os.path.exists(DEHAZE_MODEL_PATH),
        yolo_ready=YOLO_AVAILABLE,
    )

    # ── Weather ──────────────────────────────
    st.markdown('<div class="nv-section-label">Live Weather</div>', unsafe_allow_html=True)
    weather_city    = st.text_input("City", value="Dubai", label_visibility="collapsed",
                                    placeholder="Enter city (e.g. Dubai)")
    weather_api_key = st.text_input("OpenWeather API Key", value=OPENWEATHER_API_KEY,
                                    type="password", label_visibility="collapsed",
                                    placeholder="Paste OpenWeatherMap API key")

    weather_data = None
    if weather_city and weather_api_key:
        with st.spinner("Loading weather..."):
            weather_data = get_weather(weather_city, weather_api_key)
    render_weather_sidebar(weather_data, weather_city)

    st.markdown('<div class="nv-section-label">Input Mode</div>', unsafe_allow_html=True)
    app_mode = st.radio(
        "Mode",
        ['Image Upload', 'Video Upload', 'Live Camera'],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="nv-section-label">Enhancement</div>', unsafe_allow_html=True)
    strength       = st.slider('Enhancement strength',  0.0, 1.0, 1.0, 0.05)
    dcp_only       = st.toggle('DCP only mode',        value=False)
    inference_size = st.selectbox('Inference size (px)', [128, 192, 256], index=1)

    st.markdown('<div class="nv-section-label">Object Detection</div>', unsafe_allow_html=True)
    enable_detection      = st.toggle('Enable YOLO detection',   value=True)
    conf_threshold        = st.slider('Confidence threshold',  0.10, 0.90, 0.35, 0.05)
    only_driving_classes  = st.toggle('Driving classes only',    value=True)
    draw_ar_style         = st.toggle('AR-style overlay',        value=True)

    st.markdown('<div class="nv-section-label">Navigation HUD</div>', unsafe_allow_html=True)
    show_nav_hud = st.toggle('Show Nav HUD overlay', value=True)
    route_name   = st.text_input("Route name", value="Highway Demo", label_visibility="collapsed",
                                  placeholder="Route name")

    st.markdown('<div class="nv-section-label">Video Settings</div>', unsafe_allow_html=True)
    video_max_frames   = st.slider('Max frames to process', 10, 180, 60, 10)
    video_frame_skip   = st.slider('Process every Nth frame', 1, 10, 3, 1)
    split_screen_video = st.toggle('Split-screen output', value=True)

    st.markdown('<div class="nv-section-label">Live Camera</div>', unsafe_allow_html=True)
    dehaze_every = st.slider('Dehaze every N frames', 1, 6, 3, 1,
                              help="1=every frame (slow), 3=smooth ~8fps")

    st.markdown("""
    <div style="padding:16px 20px 20px;font-size:10px;font-family:'Space Mono',monospace;
    color:#2e4455;line-height:1.8;border-top:1px solid rgba(0,255,180,0.08);margin-top:12px">
    Best flow: Image for quality · Video for full pipeline demo · Analytics auto-generated after video.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DOWNLOAD & LOAD
# ─────────────────────────────────────────────
if not os.path.exists(DEHAZE_MODEL_PATH):
    with st.spinner('Downloading dehazing model from Google Drive ...'):
        ok, err = download_dehaze_model_if_needed()
    if ok:
        st.rerun()
    else:
        st.error(f'Could not download model: {err}')
        st.stop()

try:
    with st.spinner('Loading AI models ...'):
        dcp_model, resnet_model, device, missing_keys, unexpected_keys = load_dehaze_models()
        if enable_detection and YOLO_AVAILABLE:
            yolo_model = load_yolo_model()
except Exception as e:
    st.error(f'Model loading failed: {e}')
    st.stop()


# ─────────────────────────────────────────────
# PAGE: IMAGE UPLOAD
# ─────────────────────────────────────────────
if app_mode == 'Image Upload':
    topbar("NEXTGEN VISION AI / Image Enhancement")
    hero()
    metric_cards(device, enable_detection, inference_size)

    st.markdown('<div style="padding:0 36px 0">', unsafe_allow_html=True)
    section_head("Image Enhancement", "MODE: IMAGE")
    info_box("Upload a hazy or foggy road image — compare original vs dehazed, and view AR detection output.")

    uploaded = st.file_uploader(
        'Upload a hazy / foggy road image',
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        rgb = np.array(Image.open(uploaded).convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        with st.spinner('Running pipeline ...'):
            deh, final, dets, dt, yt = process_pipeline(
                bgr, strength, dcp_only, inference_size,
                enable_detection, conf_threshold,
                only_driving_classes, draw_ar_style, 'IMAGE',
                frame_idx=0, total_frames=1,
                show_nav=show_nav_hud, route_name=route_name
            )

        st.markdown('<div style="padding:0 36px">', unsafe_allow_html=True)

        split = make_split_frame(bgr, deh)
        st.markdown("""
        <div class="nv-section-head" style="margin-top:8px">
          <div class="nv-section-title" style="font-size:14px">Before / After Split</div>
          <div class="nv-section-tag">COMPARISON</div>
        </div>""", unsafe_allow_html=True)
        st.image(cv2.cvtColor(split, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("""
        <div class="nv-section-head" style="margin-top:20px">
          <div class="nv-section-title" style="font-size:14px">AR Detection + Nav HUD</div>
          <div class="nv-section-tag">LIVE AR</div>
        </div>""", unsafe_allow_html=True)
        st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True)

        oscore, _, _ = visibility_score(bgr)
        escore, _, _ = visibility_score(deh)
        vis_bars(oscore, escore, dt + yt, len(dets))

        if dets:
            st.markdown(f"""
            <div class="nv-det-table">
              <div class="nv-det-table-head">
                <div class="nv-det-title">Detection Results</div>
                <div class="nv-det-count">{len(dets)} object(s) detected</div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(dets), use_container_width=True, hide_index=True)
        else:
            st.info('No driving-related objects detected in this frame.')

        with st.expander('Technical Metrics'):
            _, oc, osh = visibility_score(bgr)
            _, ec, esh = visibility_score(deh)
            st.json({
                "original_contrast":  oc,  "enhanced_contrast":  ec,
                "original_sharpness": osh, "enhanced_sharpness": esh,
                "dehaze_seconds":     round(dt, 3),
                "detect_seconds":     round(yt, 3),
                "total_seconds":      round(dt + yt, 3),
                "inference_size":     inference_size,
                "yolo_conf":          conf_threshold,
            })

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:0 36px">
          <div style="text-align:center;padding:60px 32px;border:1px dashed rgba(0,255,180,0.2);
          border-radius:10px;background:rgba(0,255,180,0.01);margin-top:8px">
            <div style="font-size:36px;margin-bottom:16px;opacity:0.3">⬆</div>
            <div style="font-size:14px;font-weight:700;color:#eef4ff;margin-bottom:8px">No image loaded</div>
            <div style="font-size:11px;font-family:'Space Mono',monospace;color:#2e4455">
              Use the file uploader above to get started
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with st.expander('System Diagnostics'):
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:#4a6070;line-height:2">
        Dehazing missing keys: {len(missing_keys)}<br>
        Dehazing unexpected keys: {len(unexpected_keys)}<br>
        YOLO available: {YOLO_AVAILABLE}<br>
        WebRTC available: {WEBRTC_AVAILABLE}<br>
        Torch device: {device}<br>
        CUDA available: {torch.cuda.is_available()}
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: VIDEO UPLOAD
# ─────────────────────────────────────────────
elif app_mode == 'Video Upload':
    topbar("NEXTGEN VISION AI / Video + Analytics")
    hero()
    metric_cards(device, enable_detection, inference_size)

    st.markdown('<div style="padding:0 36px">', unsafe_allow_html=True)
    section_head("Video Processing", "MODE: VIDEO")
    info_box("Upload a dashcam video — get split-screen output, nav HUD overlay, and full trip analytics report.")

    uploaded_video = st.file_uploader(
        'Upload a hazy / foggy road video',
        type=['mp4', 'avi', 'mov', 'mkv'],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_video:
        inp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        inp.write(uploaded_video.read())
        inp.close()

        st.markdown('<div style="padding:0 36px">', unsafe_allow_html=True)
        st.video(inp.name)

        if st.button('⚡ PROCESS VIDEO + GENERATE ANALYTICS', use_container_width=False):
            cap = cv2.VideoCapture(inp.name)
            fps_vid = cap.get(cv2.CAP_PROP_FPS)
            fps_vid = fps_vid if fps_vid and fps_vid > 0 else 10
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            total_cap_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_proc_frames = min(video_max_frames, total_cap_frames // video_frame_skip)

            out_w    = (w * 2 + 4) if split_screen_video else w
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                max(1, fps_vid / video_frame_skip), (out_w, h)
            )

            prog    = st.progress(0)
            status  = st.empty()
            preview = st.empty()

            idx = processed = total_det = 0
            vis_history = []
            det_history = []
            class_counts = {}
            start = time.time()

            while cap.isOpened() and processed < video_max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if idx % video_frame_skip != 0:
                    continue

                deh, final, dets, dt, yt = process_pipeline(
                    frame, strength, dcp_only, inference_size,
                    enable_detection, conf_threshold,
                    only_driving_classes, draw_ar_style, 'VIDEO',
                    frame_idx=processed, total_frames=total_proc_frames,
                    show_nav=show_nav_hud, route_name=route_name
                )

                oscore, _, _ = visibility_score(frame)
                escore, _, _ = visibility_score(deh)
                vis_history.append((oscore, escore))
                det_history.append(len(dets))
                total_det += len(dets)
                for d in dets:
                    class_counts[d['class']] = class_counts.get(d['class'], 0) + 1

                write_frame = make_split_frame(frame, final) if split_screen_video else final
                out.write(write_frame)
                processed += 1

                if processed % 3 == 0:
                    preview.image(
                        cv2.cvtColor(write_frame, cv2.COLOR_BGR2RGB),
                        caption=f'Frame {processed} / {video_max_frames}  |  Objects: {len(dets)}  |  Vis: {oscore}% → {escore}%',
                        use_container_width=True,
                    )
                prog.progress(min(processed / video_max_frames, 1.0))
                status.markdown(
                    f'<div class="nv-info">Processing frame {processed} / {video_max_frames} &nbsp;·&nbsp; Detections: {len(dets)} &nbsp;·&nbsp; Visibility {oscore}% → {escore}%</div>',
                    unsafe_allow_html=True,
                )

            cap.release()
            out.release()
            total_time = time.time() - start

            st.success('Video processing complete.')
            st.video(out_path)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Frames Processed', processed)
            c2.metric('Total Time',        f'{total_time:.1f}s')
            c3.metric('Avg / Frame',       f'{total_time/max(processed,1):.2f}s')
            c4.metric('Total Detections',  total_det)

            with open(out_path, 'rb') as f:
                st.download_button(
                    '↓ DOWNLOAD PROCESSED VIDEO',
                    data=f,
                    file_name='nextgen_vision_processed.mp4',
                    mime='video/mp4',
                )

            st.markdown('<hr style="margin:32px 0">', unsafe_allow_html=True)
            analytics = {
                "frames_processed":    processed,
                "total_detections":    total_det,
                "avg_orig_visibility": np.mean([v[0] for v in vis_history]) if vis_history else 0,
                "avg_enh_visibility":  np.mean([v[1] for v in vis_history]) if vis_history else 0,
                "total_time":          total_time,
                "visibility_history":  vis_history,
                "detection_history":   det_history,
                "class_counts":        class_counts,
            }
            render_trip_analytics(analytics, weather_data)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:0 36px">
          <div style="text-align:center;padding:60px 32px;border:1px dashed rgba(0,255,180,0.2);
          border-radius:10px;background:rgba(0,255,180,0.01)">
            <div style="font-size:36px;margin-bottom:16px;opacity:0.3">🎬</div>
            <div style="font-size:14px;font-weight:700;color:#eef4ff;margin-bottom:8px">No video loaded</div>
            <div style="font-size:11px;font-family:'Space Mono',monospace;color:#2e4455">
              Upload a dashcam road video above to begin
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: LIVE CAMERA
# ─────────────────────────────────────────────
else:
    topbar("NEXTGEN VISION AI / Live Camera")
    hero()
    metric_cards(device, enable_detection, inference_size)

    st.markdown('<div style="padding:0 36px">', unsafe_allow_html=True)
    section_head("Live Camera", "MODE: LIVE")
    warn_box("Live camera uses WebRTC. If the stream doesn't connect, try a different browser or network.")

    if not WEBRTC_AVAILABLE:
        st.error('streamlit-webrtc is not installed. Add it to requirements.txt.')
        st.stop()

    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    })

    _cfg = {
        "strength":     strength,
        "dcp_only":     dcp_only,
        "size":         inference_size,
        "enable_det":   enable_detection,
        "conf":         conf_threshold,
        "driving_only": only_driving_classes,
        "ar_style":     draw_ar_style,
        "dehaze_every": dehaze_every,
        "show_nav":     show_nav_hud,
        "route_name":   route_name,
    }

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.dcp, self.resnet, self.device, _, _ = load_dehaze_models()
            self.yolo         = load_yolo_model()
            self.frame_count  = 0
            self.last_dehazed = None
            self.last_dets    = []
            self.last_time    = time.time()
            self.fps          = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format='bgr24')
            self.frame_count += 1
            cfg = _cfg

            try:
                if self.frame_count % cfg["dehaze_every"] == 0 or self.last_dehazed is None:
                    x = bgr_to_tensor(img, cfg["size"]).to(self.device)
                    with torch.no_grad():
                        dcp_out = self.dcp(x)
                        refined = dcp_out if cfg["dcp_only"] else (self.resnet(dcp_out) + 1) / 2
                    h, w = img.shape[:2]
                    self.last_dehazed = tensor_to_bgr(refined, (h, w))
                    if cfg["strength"] < 1.0:
                        self.last_dehazed = cv2.addWeighted(
                            img, 1.0 - cfg["strength"],
                            self.last_dehazed, cfg["strength"], 0
                        )

                if cfg["enable_det"] and self.yolo is not None:
                    final, self.last_dets = detect_objects_yolo(
                        self.last_dehazed.copy(),
                        cfg["conf"], cfg["driving_only"], cfg["ar_style"]
                    )
                else:
                    final          = self.last_dehazed.copy()
                    self.last_dets = []

                if cfg["show_nav"]:
                    final = draw_nav_hud(final, self.frame_count % 300, 300, cfg["route_name"])

                now = time.time()
                dt  = now - self.last_time
                self.last_time = now
                self.fps = 1.0 / dt if dt > 0 else self.fps

                final = draw_system_overlay(
                    final, mode=f"LIVE (dehaze/{cfg['dehaze_every']}f)",
                    fps=self.fps, detection_count=len(self.last_dets)
                )

            except Exception as e:
                final = img.copy()
                cv2.putText(final, f"Error: {str(e)[:60]}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(final, format='bgr24')

    webrtc_streamer(
        key='nextgen-live-camera',
        video_processor_factory=LiveProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            'video': {
                'width':     {'ideal': 640},
                'height':    {'ideal': 480},
                'frameRate': {'ideal': 10, 'max': 15},
            },
            'audio': False,
        },
        async_processing=True,
    )

    st.markdown("""
    <div class="nv-info" style="margin-top:1rem">
    ℹ &nbsp; <b>Demo tip:</b> Point the camera at a screen showing a foggy road scene.
    Adjust "Dehaze every N frames" in the sidebar to tune speed vs quality live.
    </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
