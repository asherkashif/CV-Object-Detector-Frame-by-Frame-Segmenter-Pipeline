# CV Object Detector & Frame-by-Frame Segmenter Pipeline

A computer vision pipeline for frame-by-frame video analysis using **OpenCV** and a **pretrained YOLO model**.

The system combines traditional computer vision preprocessing with deep-learning-based object detection. It applies Gaussian filtering, adaptive thresholding, contour extraction, YOLO object detection, dynamic object-count overlays, and frame-level performance measurement.

---

## Project Objective

The objective of this project is to build a practical computer vision pipeline that can:

* Process video frame by frame
* Apply Gaussian filtering for image smoothing
* Perform adaptive thresholding
* Extract contours and shapes
* Detect objects using a pretrained YOLO model
* Draw bounding boxes and object labels
* Display dynamic object counts
* Measure frame-level inference and processing performance
* Save processed video and quantitative metrics

---

## Pipeline

```text
Input Video
     |
     v
Read Frame
     |
     v
Grayscale Conversion
     |
     v
Gaussian Filtering
     |
     v
Adaptive Thresholding
     |
     v
Contour / Shape Extraction
     |
     +----------------------+
     |                      |
     v                      v
OpenCV Analysis       YOLO Detection
                            |
                            v
                    Bounding Boxes
                            |
                            v
                    Object Counting
                            |
                            +----------+
                                       |
                                       v
                              Dynamic Overlay
                                       |
                                       v
                              Frame Metrics
                                       |
                     +-----------------+----------------+
                     |                                  |
                     v                                  v
                Output Video                    Metrics CSV
```

---

## Technologies

* Python
* OpenCV
* Ultralytics YOLO
* NumPy
* Pandas
* Matplotlib

---

## Project Structure

```text
cv-object-detector/
│
├── input/
│   └── input_video.mp4
│
├── output/
│   └── detected_output.mp4
│
├── results/
│   ├── frame_metrics.csv
│   └── summary_metrics.csv
│
├── src/
│   ├── pipeline.py
│   └── analyze_results.py
│
├── visualizations/
│   ├── sample_detection.jpg
│   ├── object_count_over_time.png
│   ├── contour_count_over_time.png
│   ├── inference_time_over_time.png
│   └── processing_fps_over_time.png
│
├── whitepaper/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Computer Vision Processing

### Gaussian Filtering

The pipeline converts each video frame to grayscale and applies a Gaussian filter.

Gaussian filtering reduces small image noise and produces a smoother image for subsequent thresholding and contour analysis.

### Adaptive Thresholding

Adaptive thresholding converts the blurred grayscale frame into a binary representation.

Unlike a single global threshold, adaptive thresholding calculates local thresholds, which can be useful when lighting conditions vary across an image.

### Contour Extraction

Contours are extracted from the thresholded image using OpenCV.

Contours represent detected boundaries or shapes in the processed image. They are used as a traditional computer vision shape-analysis component of the pipeline.

Contour count is not treated as the number of detected objects because one object can produce multiple contours.

---

## YOLO Object Detection

A pretrained YOLO model is used to detect objects in the original video frames.

For every detection, the system extracts:

* Object class
* Confidence score
* Bounding-box coordinates

Example:

```text
car 0.92
person 0.87
motorcycle 0.81
```

The detected objects are displayed using bounding boxes and labels.

---

## Dynamic Object Counting

The system counts the YOLO detections independently for every frame.

The output video displays:

```text
Frame: 150
Objects: 7
Contours: 42
FPS: 3.20
```

Because the values are calculated for every frame, the displayed object count changes dynamically throughout the video.

---

## Frame-Level Metrics

The pipeline records the following metrics for every processed frame:

| Metric          | Description                        |
| --------------- | ---------------------------------- |
| Frame           | Frame number                       |
| Timestamp       | Video timestamp                    |
| Object Count    | Number of YOLO detections          |
| Contour Count   | Number of extracted contours       |
| Inference Time  | YOLO inference duration            |
| Processing Time | Complete frame processing duration |
| Processing FPS  | Approximate processing speed       |

The metrics are stored in:

```text
results/frame_metrics.csv
```

A summary is stored in:

```text
results/summary_metrics.csv
```

---

## Experimental Results

The pipeline processed **1,786 video frames**.

Observed results:

| Metric                      |     Result |
| --------------------------- | ---------: |
| Total Frames                |      1,786 |
| Average Objects / Frame     |       7.79 |
| Minimum Objects / Frame     |          2 |
| Maximum Objects / Frame     |         16 |
| Average Contours / Frame    |     229.13 |
| Average YOLO Inference Time | 0.2641 sec |
| Average Processing FPS      |       3.16 |

The first frame required substantially more processing time because model initialization can introduce startup overhead. Therefore, the first frame should not be interpreted as representative of steady-state processing performance.

---

## Visualizations

The project generates the following analysis graphs:

### Object Count Over Time

Shows how the number of YOLO detections changes throughout the video.

### Contour Count Over Time

Shows the number of contours extracted by the OpenCV preprocessing pipeline.

### YOLO Inference Time

Shows the inference time measured for individual frames.

### Processing FPS

Shows the approximate frame processing speed over the video.

---

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd cv-object-detector
```

Create and activate the Conda environment:

```bash
conda create -n cv-object-detector python=3.11 -y
conda activate cv-object-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Place the input video at:

```text
input/input_video.mp4
```

Run the detection pipeline:

```bash
python src/pipeline.py
```

Then analyze the generated metrics:

```bash
python src/analyze_results.py
```

---

## Output

After execution, the main outputs are:

```text
output/detected_output.mp4
results/frame_metrics.csv
results/summary_metrics.csv
visualizations/sample_detection.jpg
```

---

## Limitations

The current implementation performs object detection using a pretrained YOLO model without additional task-specific training.

Processing speed depends on available hardware, video resolution, model size, and inference configuration.

Contour extraction is a traditional shape-analysis technique and should not be interpreted as instance segmentation masks.

The measured FPS represents the processing speed of this pipeline rather than the original video's playback FPS.

---

## Future Improvements

Possible future improvements include:

* GPU-accelerated inference
* Model-specific fine-tuning
* Object tracking across frames
* Class-specific object counting
* Confidence threshold configuration
* Region-of-interest detection
* Actual instance segmentation using a segmentation model
* Real-time camera input
* Performance optimization and batching

---

## Conclusion

This project demonstrates a hybrid computer vision workflow combining traditional OpenCV techniques with deep-learning-based object detection.

OpenCV handles image preprocessing, adaptive thresholding, and contour extraction, while YOLO provides object detection and classification. The pipeline additionally generates dynamic overlays and frame-level performance metrics, making the system suitable for experimental video analysis and computer vision performance evaluation.
