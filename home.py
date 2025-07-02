import cv2
from pyzbar.pyzbar import decode
import streamlit as st
import numpy as np
from PIL import Image
import time

# Enhanced processing for small/angled barcodes
def ai_deblur(image):
    """Apply deblurring and enhancement techniques for better barcode detection"""
    processed_images = []
    
    # Method 1: Unsharp masking (works well for slight blur)
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    processed_images.append(unsharp)
    
    # Method 2: High contrast enhancement for small barcodes
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(image)
    processed_images.append(contrast_enhanced)
    
    # Method 3: Multiple scale processing (helps with small barcodes)
    for scale in [1.5, 2.0, 3.0]:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Apply sharpening to enlarged image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_large = cv2.filter2D(resized, -1, kernel)
        processed_images.append(sharpened_large)
    
    # Method 4: Gamma correction for better visibility
    gamma = 0.7  # Make darker areas brighter
    gamma_corrected = np.power(image / 255.0, gamma) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)
    processed_images.append(gamma_corrected)
    
    # Method 5: Morphological operations for cleaning up
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_morph)
    processed_images.append(opened)
    
    return processed_images

st.header("Scanner")

# Initialize session state
if "scanning" not in st.session_state:
    st.session_state.scanning = False
if "results" not in st.session_state:
    st.session_state.results = []
if "camera" not in st.session_state:
    st.session_state.camera = None

# Buttons for controlling scanning and results
if st.button("Clear results"):
    st.session_state.results = []

if st.button("Show results"):
    if st.session_state.results:
        st.write(st.session_state.results)
    else:
        st.write("No barcodes scanned yet.")

if st.button("Start Camera"):
    st.session_state.scanning = True

if st.button("Stop Camera"):
    st.session_state.scanning = False
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None

if st.session_state.scanning:
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)
        st.session_state.camera.set(3, 1920)
        st.session_state.camera.set(4, 1080)

    success, frame = st.session_state.camera.read()
    if success:
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flipped_display = cv2.flip(frame_display, 1)  # Flip the display
        st.image(flipped_display, caption="Live Feed - Click 'Capture & Scan' to analyze", use_column_width=True)

        capture_frame = st.button("📸 Capture & Scan Frame")
        if capture_frame:
            # Use original unflipped frame for processing to maintain barcode orientation
            frame_rgb = frame_display.copy()
            frame_rgb_2 = frame_rgb.copy()
            
            # Multiple preprocessing approaches for better detection
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            
            # Apply deblurring techniques
            deblurred_versions = ai_deblur(gray)
            
            # Your existing preprocessing methods
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Approach 2: Morphological operations
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_morph)
            
            # Approach 3: Multiple threshold methods
            resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            thresh_adaptive = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 15, 3)
            
            # Approach 4: Different blur and threshold
            blur_alt = cv2.medianBlur(gray, 3)
            thresh_otsu = cv2.threshold(blur_alt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            if thresh_adaptive.dtype != np.uint8:
                thresh_adaptive = thresh_adaptive.astype(np.uint8)

            # Try deblurred/enhanced versions first, then your existing methods
            image_to_scan = deblurred_versions + [enhanced, sharpened, morph, resized, blurred, thresh_adaptive, thresh_otsu, gray]
            barcodes_found_processed = False

            for img_version in image_to_scan:
                for info in decode(img_version):
                    barcodes_found_processed = True
                    barcode = info.data.decode("utf-8")
                    barcode_type = info.type
                    data = {"barcode": barcode, "type": barcode_type, "model": "deblurred" if img_version in deblurred_versions else "with image preparation"}

                    # Handle different scale factors for enlarged images
                    if img_version.shape[0] > gray.shape[0]:  # This is an enlarged image
                        scale_factor = img_version.shape[0] / gray.shape[0]
                    else:
                        scale_factor = 2.0 if img_version is resized else 1.0
                    top_left = (int(info.rect.left / scale_factor), int(info.rect.top / scale_factor))
                    bottom_right = (int((info.rect.left + info.rect.width) / scale_factor),
                                    int((info.rect.top + info.rect.height) / scale_factor))

                    cv2.rectangle(frame_rgb, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(frame_rgb, str(barcode), top_left,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_8)

                    if data not in st.session_state.results:
                        st.session_state.results.append(data)

                if barcodes_found_processed:
                    break

            if frame_rgb_2.dtype != np.uint8:
                frame_rgb_2 = frame_rgb_2.astype(np.uint8)

            barcodes_found_original = False
            for info in decode(frame_rgb_2):
                barcodes_found_original = True
                barcode = info.data.decode("utf-8")
                barcode_type = info.type
                data = {"barcode": barcode, "type": barcode_type, "model": "without image preparation"}

                top_left_2 = (info.rect.left, info.rect.top)
                bottom_right_2 = (info.rect.left + info.rect.width, info.rect.top + info.rect.height)

                cv2.rectangle(frame_rgb_2, top_left_2, bottom_right_2, (0, 255, 0), 2)
                cv2.putText(frame_rgb_2, str(barcode), top_left_2,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_8)

                if data not in st.session_state.results:
                    st.session_state.results.append(data)

            st.header("Captured Frame Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("With image preparation")
                st.image(frame_rgb, channels="RGB", use_column_width=True)
                if not barcodes_found_processed:
                    st.write("No barcodes detected in the processed image.")
            with col2:
                st.subheader("Without image preparation")
                st.image(frame_rgb_2, channels="RGB", use_column_width=True)
                if not barcodes_found_original:
                    st.write("No barcodes detected in the original image.")
                    
            st.success("Frame captured! Click 'Show results' to see detected barcodes.")
            # Don't rerun immediately after capture to let user see results
        else:
            # Only rerun for live feed if no capture happened
            time.sleep(0.05)
            st.rerun()
    else:
        st.error("Error, camera not available.")

else:
    st.info("Click 'Start Camera' to begin live preview")
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None