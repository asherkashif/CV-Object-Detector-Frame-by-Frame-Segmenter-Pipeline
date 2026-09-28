import cv2
import time
import pandas as pd
from ultralytics import YOLO
from pathlib import Path


# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = BASE_DIR / "input" / "input_video.mp4"
OUTPUT_VIDEO = BASE_DIR / "output" / "detected_output.mp4"
METRICS_FILE = BASE_DIR / "results" / "frame_metrics.csv"
SAMPLE_FRAME = BASE_DIR / "visualizations" / "sample_detection.jpg"


# ==============================
# CREATE DIRECTORIES
# ==============================

OUTPUT_VIDEO.parent.mkdir(exist_ok=True)
METRICS_FILE.parent.mkdir(exist_ok=True)
SAMPLE_FRAME.parent.mkdir(exist_ok=True)


# ==============================
# LOAD YOLO MODEL
# ==============================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully.")


# ==============================
# OPEN VIDEO
# ==============================

cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {INPUT_VIDEO}"
    )


# ==============================
# VIDEO INFORMATION
# ==============================

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo Information")
print("-----------------")
print(f"FPS: {fps}")
print(f"Resolution: {width} x {height}")
print(f"Total Frames: {total_frames}")


# ==============================
# OUTPUT VIDEO WRITER
# ==============================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# ==============================
# FRAME METRICS
# ==============================

metrics = []

frame_number = 0
sample_saved = False


# ==============================
# PROCESS VIDEO FRAME BY FRAME
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    frame_start = time.perf_counter()


    # ==================================
    # 1. GAUSSIAN FILTER
    # ==================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # ==================================
    # 2. ADAPTIVE THRESHOLD
    # ==================================

    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )


    # ==================================
    # 3. SHAPE / CONTOUR EXTRACTION
    # ==================================

    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_count = len(contours)


    # ==================================
    # 4. YOLO OBJECT DETECTION
    # ==================================

    inference_start = time.perf_counter()

    results = model(
        frame,
        verbose=False
    )

    inference_time = time.perf_counter() - inference_start


    # ==================================
    # 5. DRAW YOLO DETECTIONS
    # ==================================

    annotated_frame = frame.copy()

    detection_count = 0

    detected_objects = []


    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            detection_count += 1

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = model.names[class_id]

            detected_objects.append(class_name)


            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Label
            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


    # ==================================
    # 6. FRAME PROCESSING METRICS
    # ==================================

    processing_time = (
        time.perf_counter() - frame_start
    )

    processing_fps = (
        1 / processing_time
        if processing_time > 0
        else 0
    )


    # ==================================
    # 7. DYNAMIC COUNT OVERLAY
    # ==================================

    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (330, 125),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Objects: {detection_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Contours: {contour_count}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"FPS: {processing_fps:.2f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ==================================
    # 8. SAVE SAMPLE FRAME
    # ==================================

    if not sample_saved and frame_number > 30:

        cv2.imwrite(
            str(SAMPLE_FRAME),
            annotated_frame
        )

        sample_saved = True


    # ==================================
    # 9. SAVE OUTPUT VIDEO
    # ==================================

    writer.write(annotated_frame)


    # ==================================
    # 10. SAVE FRAME METRICS
    # ==================================

    metrics.append({
        "frame": frame_number,
        "timestamp_seconds": frame_number / fps,
        "object_count": detection_count,
        "contour_count": contour_count,
        "inference_time_seconds": inference_time,
        "processing_time_seconds": processing_time,
        "processing_fps": processing_fps
    })


    # ==================================
    # PROGRESS
    # ==================================

    if frame_number % 30 == 0:

        print(
            f"Processed frame {frame_number}/{total_frames}"
        )


# ==============================
# RELEASE RESOURCES
# ==============================

cap.release()
writer.release()


# ==============================
# SAVE METRICS CSV
# ==============================

df = pd.DataFrame(metrics)

df.to_csv(
    METRICS_FILE,
    index=False
)


# ==============================
# FINAL SUMMARY
# ==============================

print("\n================================")
print("PIPELINE COMPLETED")
print("================================")

print(f"Processed frames: {frame_number}")

print(f"Output video:")
print(OUTPUT_VIDEO)

print(f"\nFrame metrics:")
print(METRICS_FILE)

print(f"\nSample detection frame:")
print(SAMPLE_FRAME)

print("\nAverage objects per frame:")
print(f"{df['object_count'].mean():.2f}")

print("\nAverage inference time:")
print(
    f"{df['inference_time_seconds'].mean():.4f} seconds"
)

print("\nAverage processing FPS:")
print(
    f"{df['processing_fps'].mean():.2f}"
)