# VisionAudit: Classical Preprocessing, Feature Geometry & Object Classification Pipeline

A modular, headless computer vision tool designed for automated object inspection, boundary extraction, and machine learning classification.

## Features
- **Deterministic Preprocessing:** Grayscale conversion, median filtering for noise removal, and global histogram equalization.
- **Geometric Primitive Extraction:** Canny edge maps, Harris corner response detection, and Hough probabilistic line transforms.
- **Morphological Segmentation:** Foreground extraction using Otsu thresholding, morphological opening/closing, and contour bounding-box cropping.
- **Classification Engine:** Patch feature vector compression via Principal Component Analysis (PCA) and K-Nearest Neighbors (KNN) inference.
- **Headless Execution:** Designed for pure terminal environments without GUI display dependencies.

## Technologies Used
- Python 3.10+
- OpenCV (`opencv-python-headless`)
- NumPy
- Scikit-learn
- Matplotlib

## Environment Setup & Installation

1. Clone the repository:
```bash
git clone [https://github.com/](https://github.com/)<your-github-username>/vision-audit.git
cd vision-audit
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

Run the pipeline by passing any image file path via the `--image` argument:

```bash
python main.py --image sample.jpg --outdir output
```

You can pass arbitrary custom images from any location:
```bash
python main.py --image /path/to/any_image.png --outdir output
```

## Output Artifacts
The script automatically generates an `output/` directory containing:
- `annotated_result.png`: Input image with localized bounding boxes and predicted class tags.
- `pipeline_summary.png`: A 4-panel diagnostic plot displaying original input, equalized grayscale, Canny edge detection, and classified bounding boxes.

## Testing Instructions
Run pipeline sanity tests to verify module execution and shape integrity:
```bash
python -m unittest discover -s . -p "test_*.py"
```
