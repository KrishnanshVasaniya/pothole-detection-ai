import cv2
import numpy as np
import pandas as pd

def find_pothole_regions(image_path):
    from PIL import Image

    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 2000:  # tweak size for potholes
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    return image, edges, boxes

def detect_potholes_combined(lidar_csv, image_path):
    lidar = pd.read_csv(lidar_csv)
    image, edges, boxes = find_pothole_regions(image_path)

    # Simulate depth-based pothole detection
    pothole_mask = lidar['z'] < -0.05
    pothole_depths = lidar['z'][pothole_mask]
    avg_depth = round(pothole_depths.mean(), 4) if not pothole_depths.empty else 0.0

    return image, boxes, avg_depth, lidar, pothole_mask, edges
