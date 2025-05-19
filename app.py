import streamlit as st
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
from detection import detect_potholes_combined
from simulate_lidar import generate_lidar_data

st.set_page_config(page_title="Pothole Detection", layout="wide")
st.title("🚧 Pothole Detection System (Live Upload + Batch Mode)")

# Generate LiDAR data if missing
if not os.path.exists("lidar_data.csv"):
    generate_lidar_data()

# Upload images
uploaded_files = st.file_uploader("Upload road images (JPG/PNG)", type=['jpg', 'png'], accept_multiple_files=True)

if not uploaded_files:
    st.info("No image uploaded. Showing default image.")
    uploaded_files = [open("road_image.jpg", "rb")]

# Show each uploaded image result
for uploaded_file in uploaded_files:
    st.markdown("---")
    st.subheader(f"🖼️ Processing: {uploaded_file.name if hasattr(uploaded_file, 'name') else 'Default Image'}")

    # Save the uploaded file to disk temporarily
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    image, boxes, avg_depth, lidar, potholes, edges = detect_potholes_combined("lidar_data.csv", image_path)

    # Calculate "road condition percentage" based on boxes vs image size
    img_area = image.shape[0] * image.shape[1]
    pothole_area = sum([w * h for (_, _, w, h) in boxes])
    road_quality = max(0, 100 - int((pothole_area / img_area) * 100))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📡 LiDAR Depth")
        fig, ax = plt.subplots()
        ax.plot(lidar['x'], lidar['z'])
        ax.scatter(lidar['x'][potholes], lidar['z'][potholes], color='red')
        ax.set_xlabel("X")
        ax.set_ylabel("Depth (Z)")
        st.pyplot(fig)

    with col2:
        st.markdown("### 📷 Canny Edge Detection")
        st.image(edges, channels="GRAY", use_column_width=True)

    with col3:
        st.markdown("### 📦 Potholes (Bounding Boxes)")
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(image, channels="BGR", use_column_width=True)

    st.success(f"✅ Average pothole depth: {avg_depth} meters")
    st.info(f"🏁 Estimated Road Condition: {road_quality}%")

