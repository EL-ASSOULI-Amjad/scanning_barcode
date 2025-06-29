import cv2
from pyzbar.pyzbar import decode
import streamlit as st
import numpy as np
from PIL import Image

st.header("Scanner")

# Initialize session state
if "scanning" not in st.session_state:
    st.session_state.scanning = False
if "results" not in st.session_state:
    st.session_state.results = []

# Buttons for controlling scanning and results
if st.button("Clear results"):
    st.session_state.results = []

if st.button("Show results"):
    if st.session_state.results:
        st.write(st.session_state.results)
    else:
        st.write("No barcodes scanned yet.")

# Camera input for capturing images
camera_image = st.file_uploader("Upload the image with the barcode", type=["jpg", "png"], accept_multiple_files=False, key=None)

# Process the image when captured
if camera_image is not None:
    # Convert the uploaded image to a format OpenCV can process
    
    frame_rgb = np.array(camera_image)  # Convert PIL image to numpy array (RGB)

    # Decode barcodes in the image
    barcodes_found = False
    for info in decode(frame_rgb):
        barcodes_found = True
        barcode = info.data.decode("utf-8")  # Decode bytes to string
        barcode_type = info.type
        data = {"barcode": barcode, "type": barcode_type}

        # Get rectangle coordinates for visualization
        top_left = (info.rect.left, info.rect.top)
        bottom_right = (info.rect.left + info.rect.width, info.rect.top + info.rect.height)

        # Draw rectangle and text on the image
        cv2.rectangle(frame_rgb, top_left, bottom_right, color=(0, 255, 0), thickness=2)
        cv2.putText(
            frame_rgb,
            str(barcode),
            top_left,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
            cv2.LINE_8,
        )

        # Append barcode data to results if not already present
        if data not in st.session_state.results:
            st.session_state.results.append(data)

    # Display the image with barcode annotations (or original if no barcodes)
    st.image(frame_rgb, channels="RGB", use_column_width=True)
    if not barcodes_found:
        st.write("No barcodes detected in the image.")