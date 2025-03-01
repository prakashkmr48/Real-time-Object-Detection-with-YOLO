import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize session state variables
if 'captured_frames' not in st.session_state:
    st.session_state['captured_frames'] = []
if 'run' not in st.session_state:
    st.session_state['run'] = False
if 'capture' not in st.session_state:
    st.session_state['capture'] = False

def process_frame(frame):
    """
    Performs object detection on a frame and returns the annotated frame.
    """
    results = model(frame)
    annotated_frame = frame.copy() # Make a copy to avoid modifying the original
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

    return annotated_frame

def main():
    st.title("Real-time Object Detection with YOLOv8")
    st.write("Click 'Start/Capture' to begin object detection and capture frames.")

    # Create a placeholder for the video feed
    video_placeholder = st.empty()

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state['run']:
            button_label = "Capture"
        else:
            button_label = "Start"
        start_capture_button = st.button(button_label, on_click=start_capture_clicked)
    with col2:
        clear_button = st.button("Clear", on_click=clear_clicked)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while st.session_state['run']:
        ret, frame = cap.read()
        if not ret:
            st.write("Video capture ended.")
            break

        # Process the frame
        annotated_frame = process_frame(frame)

        # Display the annotated frame in Streamlit
        video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        if st.session_state['capture']:
            st.session_state['captured_frames'].append(annotated_frame)
            st.session_state['capture'] = False  # Reset capture flag

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

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

def start_capture_clicked():
    if st.session_state['run']:
        # If running, capture a frame
        st.session_state['capture'] = True
    else:
        # If not running, start the video stream
        st.session_state['run'] = True

def clear_clicked():
    st.session_state['captured_frames'] = []
    st.session_state['run'] = False

if __name__ == "__main__":
    main()
