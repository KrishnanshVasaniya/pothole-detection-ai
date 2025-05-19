import os
import cv2
import pandas as pd
from detection import detect_potholes_combined

INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "output"
LIDAR_CSV = "lidar_data.csv"  # Use same simulated LiDAR for all or create per image

results = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(INPUT_FOLDER, filename)
        image, boxes, avg_depth, _, _, _ = detect_potholes_combined(LIDAR_CSV, image_path)

        # Draw boxes and save
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, image)

        # Save results to CSV
        results.append({
            "filename": filename,
            "num_potholes": len(boxes),
            "avg_depth": avg_depth
        })

# Save detection summary
df = pd.DataFrame(results)
df.to_csv("pothole_results_summary.csv", index=False)
print("✅ Batch detection complete. Check 'output/' and summary CSV.")
