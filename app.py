import os

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from backend.frame_processor import FrameProcessor

def get_secret_or_env(name):
    #-------------- Prefer Streamlit secrets, then fall back to environment variables -----------------
    try:
        return st.secrets.get(name, os.getenv(name))
    except FileNotFoundError:
        return os.getenv(name)


def get_rtc_configuration():
    #-------------- Use TURN credentials from Streamlit secrets or environment variables -----------------
    turn_username = get_secret_or_env("TURN_USERNAME")
    turn_credential = get_secret_or_env("TURN_CREDENTIAL")

    ice_servers = [
        {"urls": ["stun:stun.relay.metered.ca:80", "stun:stun.l.google.com:19302"]}
    ]

    if turn_username and turn_credential:
        turn_servers = [
            "turn:global.relay.metered.ca:80",
            "turn:global.relay.metered.ca:80?transport=tcp",
            "turn:global.relay.metered.ca:443",
            "turns:global.relay.metered.ca:443?transport=tcp",
        ]

        ice_servers.extend(
            {
                "urls": server,
                "username": turn_username,
                "credential": turn_credential,
            }
            for server in turn_servers
        )

    return {"iceServers": ice_servers}

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Air Writing Recognition",
    page_icon="✍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------
# Load CSS
# --------------------------------
with open("style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True,
    )

# --------------------------------
# Sidebar
# --------------------------------
with st.sidebar:
    #-------------- How to use instructions header -----------------
    st.header("ℹ️ How to Use")

    #-------------- Gesture controls card -----------------
    st.markdown(
        """
        <div class="instructions">
            <h4>🖐 Gesture Controls</h4>
            <ul>
                <li>☝️ <b>Index Finger</b> → Draw</li>
                <li>☝️☝️☝️ <b>Index, Middle, Ring Fingers</b> → Predict</li>
                <li>✋ <b>Open Palm</b> → Clear Canvas</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    #-------------- Model info card -----------------
    st.markdown(
        """
        <div class="model-card">
            <div class="Model-title">
                🎯 Model
            </div>
            <div class="model-Name">
                KNN
            </div>
            <div class="Features">
                HOG Features
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    #-------------- Project information card -----------------
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">
                📊 Project Information
            </div>
            <div class="info-text">
                • Model : K-Nearest Neighbors<br>
                • Feature Extraction : HOG<br>
                • Hand Tracking : MediaPipe Tasks<br>
                • Input Size : 34 × 34 Pixels
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    

# --------------------------------
# Header
# --------------------------------
#-------------- Main title and subtitle card -----------------
st.markdown(
    """
    <div class="main-header">
        <h1>✍ Air Writing Recognition</h1>
        <p>Real-Time English Alphabet Recognition using MediaPipe + Machine Learning (KNN)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="model-disclaimer">
        ⚠️ <b>Note:</b> This model may occasionally predict the wrong letter,
        especially with unclear gestures or poor lighting. For best results,
        draw slowly and clearly.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
# --------------------------------
# Live Camera (Centered, Fixed Size)
# --------------------------------
st.subheader("📷 Live Camera")

#-------------- Bordered container holding the webcam stream -----------------
with st.container(border=True):
    #-------------- Start webcam stream and connect it to the frame processor -----------------
    webrtc_streamer(
    key="air-writing",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FrameProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15},
            "width": {"ideal": 320},
            "height": {"ideal": 240},
            "frameRate": {"ideal": 10, "max": 10},
        },
        "audio": False,
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    rtc_configuration=get_rtc_configuration(),
    async_processing=True,
    video_html_attrs={
        "style": {
            "width": "100%",
            "max-width": "600px",
            "height": "450px",
            "object-fit": "cover",
            "margin": "0 auto",
            "border": "2px solid #38bdf8",
            "border-radius": "12px",
            "display": "block",
        },
        "controls": False,
        "autoPlay": True,
        "playsInline": True,
        "muted": True,
    },
)
)
