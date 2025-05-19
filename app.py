import streamlit as st
import matplotlib.pyplot as plt
import cv2
from detection import detect_potholes_combined
from PIL import Image

st.set_page_config(page_title="Pothole Detection", layout="centered")
st.title("🚧 Simulated Pothole Detection System")
st.markdown("Fusing LiDAR + Camera to detect potholes using depth + edge analysis (Simulated)")

# Run the upgraded detection
image, boxes, avg_depth, lidar, potholes, edges = detect_potholes_combined("lidar_data.csv", "road_image.jpg")

# 📡 LiDAR Visualization
st.subheader("📡 LiDAR Depth Visualization")
fig, ax = plt.subplots()
ax.plot(lidar['x'], lidar['z'], label="Depth")
ax.scatter(lidar['x'][potholes], lidar['z'][potholes], color='red', label="Potholes")
ax.set_xlabel("Distance (X)")
ax.set_ylabel("Depth (Z)")
ax.legend()
st.pyplot(fig)

# 📷 Edge Detection Output
st.subheader("📷 Road Image Edges (Canny)")
st.image(edges, channels="GRAY", caption="Canny Edge Detection")

# 📦 Draw Bounding Boxes on Image
st.subheader("📦 Detected Pothole Regions (Bounding Boxes)")
img_copy = image.copy()
for (x, y, w, h) in boxes:
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

st.image(img_copy, channels="BGR", caption="Potholes Detected with Bounding Boxes")

# 📏 Average Depth Display
st.subheader("📏 Average Pothole Depth from LiDAR")
if avg_depth > 0:
    st.success(f"Estimated Average Pothole Depth: **{avg_depth} meters**")
else:
    st.info("No potholes detected below depth threshold.")
