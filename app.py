import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image
import os
from twilio.rest import Client

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize session state variables
if 'captured_frames' not in st.session_state:
    st.session_state['captured_frames'] = []
if 'capture' not in st.session_state:
    st.session_state['capture'] = False

def process_frame(frame):
    """
    Performs object detection on a frame and returns the annotated frame.
    """
    results = model(frame)
    annotated_frame = frame.copy()  # Make a copy to avoid modifying the original
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            conf = box.conf[0]
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    annotated_av_frame = process_frame(img)
    if st.session_state['capture']:
        st.session_state['captured_frames'].append(annotated_av_frame.to_ndarray(format="bgr24"))
        st.session_state['capture'] = False
    return annotated_av_frame

def main():
    st.title("Real-time Object Detection with YOLOv8")
    st.write("Click 'Capture' to capture frames.")

    # Twilio configuration
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]

    if account_sid and auth_token:
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        rtc_configuration = {"iceServers": token.ice_servers}
    else:
        rtc_configuration = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
        st.warning("Twilio credentials not found. Using default STUN server. TURN might be needed.")

    webrtc_streamer(key="example", video_frame_callback=video_frame_callback, rtc_configuration=rtc_configuration)

    col1, col2 = st.columns(2)
    with col1:
        start_capture_button = st.button("Capture", on_click=start_capture_clicked)
    with col2:
        clear_button = st.button("Clear", on_click=clear_clicked)

    # Display captured frames
    if st.session_state['captured_frames']:
        st.write("## Captured Frames:")
        for i, captured_frame in enumerate(st.session_state['captured_frames']):
            st.image(captured_frame, channels="BGR", caption=f"Captured {i+1}", use_column_width=True)

    # Download button for the last captured image
    if st.session_state['captured_frames']:
        last_frame = st.session_state['captured_frames'][-1]
        pil_img = Image.fromarray(last_frame.astype('uint8'), 'BGR')
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        st.download_button(
            label="Download Last Captured Image",
            data=img_bytes,
            file_name="captured_image.png",
            mime="image/png",
        )

def start_capture_clicked():
    st.session_state['capture'] = True

def clear_clicked():
    st.session_state['captured_frames'] = []

if __name__ == "__main__":
    main()
