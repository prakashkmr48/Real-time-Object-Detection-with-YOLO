import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import asyncio
import numpy as np
import io
from PIL import Image
import os
from threading import Thread
from ultralytics import YOLO
from twilio.rest import Client

# Fix asyncio issue in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Disable Streamlit's auto-reload to avoid __path__._path error
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize session state
if 'captured_frames' not in st.session_state:
    st.session_state['captured_frames'] = []
if 'capture' not in st.session_state:
    st.session_state['capture'] = False

# Function to fetch Twilio ICE servers (run in a separate thread)
def fetch_twilio_servers():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning(f"Twilio Error: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# Get ICE configuration in a background thread
twilio_thread = Thread(target=fetch_twilio_servers)
twilio_thread.start()
twilio_thread.join()
rtc_configuration = {"iceServers": fetch_twilio_servers()}

# Object detection function
def process_frame(frame):
    results = model(frame)
    annotated_frame = frame.copy()
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            conf = box.conf[0]
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    annotated_av_frame = process_frame(img)
    if st.session_state['capture']:
        st.session_state['captured_frames'].append(annotated_av_frame.to_ndarray(format="bgr24"))
        st.session_state['capture'] = False
    return annotated_av_frame

# Streamlit UI
def main():
    st.title("Real-time Object Detection with YOLOv8")
    st.write("Click 'Capture' to save frames.")

    # WebRTC streamer
    webrtc_streamer(key="example", video_frame_callback=video_frame_callback, rtc_configuration=rtc_configuration)

    # Buttons for capture and clearing frames
    col1, col2 = st.columns(2)
    with col1:
        st.button("Capture", on_click=lambda: st.session_state.update({"capture": True}))
    with col2:
        st.button("Clear", on_click=lambda: st.session_state.update({"captured_frames": []}))

    # Display captured frames
    if st.session_state['captured_frames']:
        st.write("## Captured Frames:")
        for i, captured_frame in enumerate(st.session_state['captured_frames']):
            st.image(captured_frame, channels="BGR", caption=f"Captured {i+1}", use_column_width=True)

    # Download last captured image
    if st.session_state['captured_frames']:
        last_frame = st.session_state['captured_frames'][-1]
        pil_img = Image.fromarray(last_frame.astype('uint8'), 'BGR')
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        st.download_button(label="Download Last Captured Image", data=buf.getvalue(),
                           file_name="captured_image.png", mime="image/png")

if __name__ == "__main__":
    main()
