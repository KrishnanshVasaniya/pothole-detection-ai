import numpy as np
import pandas as pd

def generate_lidar_data(num_points=500):
    x = np.linspace(0, 10, num_points)
    y = np.random.normal(0, 0.01, num_points)
    z = np.random.normal(0, 0.01, num_points)

    # Simulate potholes (depth drop)
    pothole_indices = np.random.choice(num_points, size=5, replace=False)
    z[pothole_indices] -= np.random.uniform(0.1, 0.3, size=5)

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    df.to_csv('lidar_data.csv', index=False)
    print("✅ LiDAR data generated and saved to lidar_data.csv")

if __name__ == "__main__":
    generate_lidar_data()
