#!/usr/bin/env python3
import sys
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firevision.video.config import load_config
from firevision.video.detector import UltralyticsDetector
from firevision.video.pipeline import VideoFusionEngine


st.set_page_config(page_title="FireSmoke CV Dashboard", page_icon="🔥", layout="wide")

# Sidebar Controls
st.sidebar.title("🔥 FireSmoke CV")
st.sidebar.markdown("Configure the pipeline before uploading media.")

config_path = st.sidebar.text_input("Config Path", "configs/video.yaml")
yolo_conf = st.sidebar.slider("YOLO Confidence", 0.0, 1.0, 0.25)
nms_iou = st.sidebar.slider("NMS IoU", 0.0, 1.0, 0.45)

uploaded_file = st.sidebar.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4", "avi"])

@st.cache_resource
def get_engine(cfg_path, conf, iou):
    config = load_config(cfg_path)
    # Override config with slider values
    config.detector.confidence_grid = [conf]
    config.detector.iou_grid = [iou]
    detector = UltralyticsDetector(config)
    return VideoFusionEngine(config, detector)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()
    
    st.title("Live Processing Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Media Stream")
        canvas = st.empty()
    with col2:
        st.subheader("Live Analytics")
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
    
    try:
        engine = get_engine(config_path, yolo_conf, nms_iou)
        
        if file_name.endswith((".jpg", ".png", ".jpeg")):
            # Process Single Image
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            annotated, fused_tracks = engine.process_frame(frame, 0, 0.0)
            
            # Convert BGR to RGB for Streamlit
            canvas.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            
            with metrics_placeholder.container():
                st.metric("Total Detections", len(fused_tracks))
                max_risk = max([t.risk for t in fused_tracks]) if fused_tracks else 0.0
                st.metric("Peak Risk", f"{max_risk:.2f}")
                
        elif file_name.endswith((".mp4", ".avi")):
            # Process Video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(file_bytes)
                temp_path = tfile.name
                
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0: fps = 30.0
            
            frame_idx = 0
            risk_history = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                timestamp = frame_idx / fps
                annotated, fused_tracks = engine.process_frame(frame, frame_idx, timestamp)
                
                # Convert to RGB
                canvas.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                
                max_risk = max([t.risk for t in fused_tracks]) if fused_tracks else 0.0
                risk_history.append(max_risk)
                
                # Update Analytics
                with metrics_placeholder.container():
                    st.metric("Active Tracks", len(fused_tracks))
                    st.metric("Instantaneous Risk", f"{max_risk:.2f}")
                    
                # We show the last 100 frames of history on the chart
                df = pd.DataFrame({"Max Risk Score": risk_history[-100:]})
                chart_placeholder.line_chart(df)
                
                frame_idx += 1
                
            cap.release()
            st.success("Video processing complete!")
            
    except Exception as e:
        st.error(f"Error processing media: {str(e)}")
else:
    st.title("Waiting for media...")
    st.info("Please upload a file in the sidebar to begin.")
