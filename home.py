import cv2
from pyzbar.pyzbar import decode
import streamlit as st
import numpy as np
st.header("Scanner")

if "scnanning" not in st.session_state:
    st.session_state.scanning = False

if "results" not in st.session_state:
    st.session_state.results = []

if st.button("Start scanning"):
    st.session_state.scanning = True
    st.session_state.results = []

if st.button("Show results"):
    st.write(st.session_state.results)

if st.button("Stop"):
    st.session_state.scanning = False

frame_placeholder = st.empty()

if st.session_state.scanning:
    cam = cv2.VideoCapture(0)   
    while st.session_state.scanning:
        success, frame = cam.read()
        if not success:
            print("Turn on your camera")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for info in decode(frame):
            barcode = info.data     
            type = info.type
            data = {"barcode": barcode, "type": type}
            # Get rectangle coordinates from info.rect
            top_left = (info.rect.left, info.rect.top)
            bottom_right = (info.rect.left + info.rect.width, info.rect.top + info.rect.height)
            # Draw rectangle on frame_rgb
            cv2.rectangle(frame_rgb, top_left, bottom_right, color=(0, 255, 0), thickness=2)
            cv2.putText(frame_rgb, str(barcode), top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_8)
            if data not in st.session_state.results:
                st.session_state.results.append(data)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        cv2.waitKey(1)
        if not st.session_state.scanning:
            break






    
