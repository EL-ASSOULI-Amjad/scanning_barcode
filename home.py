import cv2
from pyzbar.pyzbar import decode
import streamlit as st
import numpy as np
from PIL import Image

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

def process_image_for_barcodes(image):
    """Process uploaded image for barcode detection"""
    # Convert PIL image to OpenCV format
    frame_rgb = np.array(image)
    if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 3:
        # RGB image
        pass
    elif len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 4:
        # RGBA image, convert to RGB
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
    
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
    results = []

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

            if data not in results:
                results.append(data)

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

        if data not in results:
            results.append(data)
    
    return frame_rgb, frame_rgb_2, results, barcodes_found_processed, barcodes_found_original

st.header("Scanner")

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = []

# Buttons for controlling results
if st.button("Clear results"):
    st.session_state.results = []

if st.button("Show results"):
    if st.session_state.results:
        st.write("**Detected Barcodes:**")
        for i, result in enumerate(st.session_state.results, 1):
            st.write(f"{i}. **{result['barcode']}** (Type: {result['type']}, Method: {result['model']})")
    else:
        st.write("No barcodes scanned yet.")

st.info("📱 **Better Focus Solution:** Use your phone's regular camera app, then upload the photo below")

st.markdown("**Why this works better:**")
st.markdown("- ✅ Your phone's camera app has perfect autofocus")
st.markdown("- ✅ You can tap to focus and see it working")  
st.markdown("- ✅ Take multiple shots until you get a clear one")
st.markdown("- ✅ No browser camera limitations")

# Main camera option: Upload from phone camera
st.header("📸 Upload Barcode Photo")
camera_image = st.camera_input("Take a picture of a barcode with your phone camera")

if camera_image is not None:
    # Convert the uploaded file to PIL Image
    image = Image.open(camera_image)
    
    # Process the image for barcode detection
    with st.spinner("Processing image for barcodes..."):
        frame_processed, frame_original, new_results, found_processed, found_original = process_image_for_barcodes(image)
        
        # Add new results to session state
        for result in new_results:
            if result not in st.session_state.results:
                st.session_state.results.append(result)
    
    # Display results
    st.header("Image Analysis Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("With image preparation")
        st.image(frame_processed, channels="RGB", use_column_width=True)
        if not found_processed:
            st.write("No barcodes detected in the processed image.")
            
    with col2:
        st.subheader("Without image preparation")
        st.image(frame_original, channels="RGB", use_column_width=True)
        if not found_original:
            st.write("No barcodes detected in the original image.")
    
    if new_results:
        st.success(f"Found {len(new_results)} barcode(s)! Click 'Show results' to see all detected barcodes.")
        st.write("**Newly detected:**")
        for result in new_results:
            st.write(f"• **{result['barcode']}** (Type: {result['type']}, Method: {result['model']})")
    else:
        st.warning("No barcodes detected in this image. Try adjusting lighting or angle.")

# Option to upload image file instead
st.header("📁 Recommended for Mobile: Upload Photo")
st.markdown("*Take a photo with your regular camera app, then upload it here for better focus quality*")
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Convert the uploaded file to PIL Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image for barcode detection
    with st.spinner("Processing uploaded image for barcodes..."):
        frame_processed, frame_original, new_results, found_processed, found_original = process_image_for_barcodes(image)
        
        # Add new results to session state
        for result in new_results:
            if result not in st.session_state.results:
                st.session_state.results.append(result)
    
    # Display results
    st.header("Uploaded Image Analysis Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("With image preparation")
        st.image(frame_processed, channels="RGB", use_column_width=True)
        if not found_processed:
            st.write("No barcodes detected in the processed image.")
            
    with col2:
        st.subheader("Without image preparation")
        st.image(frame_original, channels="RGB", use_column_width=True)
        if not found_original:
            st.write("No barcodes detected in the original image.")
    
    if new_results:
        st.success(f"Found {len(new_results)} barcode(s)! Click 'Show results' to see all detected barcodes.")
        st.write("**Newly detected:**")
        for result in new_results:
            st.write(f"• **{result['barcode']}** (Type: {result['type']}, Method: {result['model']})")
    else:
        st.warning("No barcodes detected in this image. Try a different image with better lighting or angle.")