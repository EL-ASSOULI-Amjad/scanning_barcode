import cv2
from pyzbar.pyzbar import decode
import streamlit as st
import numpy as np
from PIL import Image

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

camera_image = st.camera_input("Take a picture")
if st.button("Start scanning"):
    if camera_image is not None:
        img = Image.open(camera_image)
        frame_rgb = np.array(img)   
        for info in decode(frame_rgb):
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

    






    
