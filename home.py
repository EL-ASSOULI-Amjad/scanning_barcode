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
    img = Image.open(camera_image)
    frame_rgb = np.array(img.convert("RGB"))
    # FIX 1: Change COLOR_BGR2GRAY to COLOR_RGB2GRAY since PIL gives RGB format
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if thresh.dtype != np.uint8:
        thresh = thresh.astype(np.uint8)

    # FIX 2: Try decoding on multiple image versions, not just the heavily processed one
    images_to_try = [gray, resized, blurred, thresh]
    
    # Decode barcodes in the image
    barcodes_found = False
    for img_version in images_to_try:
        for info in decode(img_version):
            barcodes_found = True
            barcode = info.data.decode("utf-8")  # Decode bytes to string
            barcode_type = info.type
            data = {"barcode": barcode, "type": barcode_type}

            # FIX 3: Draw on the original RGB image, not the processed grayscale
            # Scale coordinates back to original image size if using resized version
            scale_factor = 2.0 if img_version is resized else 1.0
            top_left = (int(info.rect.left / scale_factor), int(info.rect.top / scale_factor))
            bottom_right = (int((info.rect.left + info.rect.width) / scale_factor), 
                          int((info.rect.top + info.rect.height) / scale_factor))

            # Draw rectangle and text on the RGB image
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
        
        # If barcodes found, break out of the loop
        if barcodes_found:
            break

    # Display the image with barcode annotations (or original if no barcodes)
    st.image(frame_rgb, channels="RGB", use_column_width=True)
    if not barcodes_found:
        st.write("No barcodes detected in the image.")