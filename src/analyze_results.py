import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

METRICS_FILE = BASE_DIR / "results" / "frame_metrics.csv"
VISUALIZATION_DIR = BASE_DIR / "visualizations"

VISUALIZATION_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD METRICS
# ==========================================

df = pd.read_csv(METRICS_FILE)

print("\n================================")
print("FRAME METRICS ANALYSIS")
print("================================")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# BASIC STATISTICS
# ==========================================

print("\n================================")
print("BASIC STATISTICS")
print("================================")

print("\nAverage object count:")
print(f"{df['object_count'].mean():.2f}")

print("\nMaximum object count:")
print(df['object_count'].max())

print("\nMinimum object count:")
print(df['object_count'].min())

print("\nAverage contour count:")
print(f"{df['contour_count'].mean():.2f}")

print("\nAverage YOLO inference time:")
print(
    f"{df['inference_time_seconds'].mean():.4f} seconds"
)

print("\nAverage processing time:")
print(
    f"{df['processing_time_seconds'].mean():.4f} seconds"
)

print("\nAverage processing FPS:")
print(
    f"{df['processing_fps'].mean():.2f}"
)


# ==========================================
# OBJECT COUNT OVER TIME
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestamp_seconds"],
    df["object_count"]
)

plt.title("Object Count Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Number of Detected Objects")

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "object_count_over_time.png",
    dpi=300
)

plt.close()


# ==========================================
# CONTOUR COUNT OVER TIME
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestamp_seconds"],
    df["contour_count"]
)

plt.title("Contour Count Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Number of Contours")

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "contour_count_over_time.png",
    dpi=300
)

plt.close()


# ==========================================
# YOLO INFERENCE TIME
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestamp_seconds"],
    df["inference_time_seconds"]
)

plt.title("YOLO Inference Time Per Frame")
plt.xlabel("Time (seconds)")
plt.ylabel("Inference Time (seconds)")

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "inference_time_over_time.png",
    dpi=300
)

plt.close()


# ==========================================
# PROCESSING FPS
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["timestamp_seconds"],
    df["processing_fps"]
)

plt.title("Frame Processing FPS")
plt.xlabel("Time (seconds)")
plt.ylabel("FPS")

plt.tight_layout()

plt.savefig(
    VISUALIZATION_DIR / "processing_fps_over_time.png",
    dpi=300
)

plt.close()


# ==========================================
# SAVE SUMMARY
# ==========================================

summary = {
    "total_frames": len(df),
    "average_objects_per_frame": df["object_count"].mean(),
    "maximum_objects_per_frame": df["object_count"].max(),
    "minimum_objects_per_frame": df["object_count"].min(),
    "average_contours_per_frame": df["contour_count"].mean(),
    "average_inference_time_seconds": df[
        "inference_time_seconds"
    ].mean(),
    "average_processing_time_seconds": df[
        "processing_time_seconds"
    ].mean(),
    "average_processing_fps": df[
        "processing_fps"
    ].mean()
}


summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    BASE_DIR / "results" / "summary_metrics.csv",
    index=False
)


# ==========================================
# COMPLETE
# ==========================================

print("\n================================")
print("ANALYSIS COMPLETED")
print("================================")

print("\nGenerated graphs:")

print(
    VISUALIZATION_DIR /
    "object_count_over_time.png"
)

print(
    VISUALIZATION_DIR /
    "contour_count_over_time.png"
)

print(
    VISUALIZATION_DIR /
    "inference_time_over_time.png"
)

print(
    VISUALIZATION_DIR /
    "processing_fps_over_time.png"
)

print("\nSummary:")
print(
    BASE_DIR /
    "results" /
    "summary_metrics.csv"
)