import argparse
import os
import cv2
import matplotlib.pyplot as plt
from preprocess import ImagePreprocessor
from geometry import GeometricAnalyzer
from segment import Segmenter
from classify import PatchClassifier

def run_pipeline(input_image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Loading and Preprocessing: {input_image_path}")

    # 1. Preprocess
    preprocessor = ImagePreprocessor()
    prep_res = preprocessor.process(input_image_path)
    enhanced_gray = prep_res["enhanced"]
    original_bgr = prep_res["original_bgr"]

    # 2. Geometric Analysis
    print("[+] Extracting Edges, Corners, and Lines...")
    geo_analyzer = GeometricAnalyzer()
    geo_res = geo_analyzer.analyze(enhanced_gray)

    # 3. Segmentation & Extraction
    print("[+] Segmenting Objects via Morphological Contours...")
    segmenter = Segmenter()
    objects = segmenter.extract_bounding_boxes(original_bgr)

    # 4. Classification via PCA + KNN
    print(f"[+] Found {len(objects)} object candidates. Classifying...")
    classifier = PatchClassifier()
    classifier.fit_synthetic_baseline()

    annotated = original_bgr.copy()
    for idx, obj in enumerate(objects):
        x, y, w, h = obj["box"]
        label = classifier.predict(obj["crop"])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"ID:{idx+1} {label}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 5. Render Headless Diagnostic Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(prep_res["enhanced"], cmap="gray")
    axes[0, 1].set_title("Denoised & Equalized")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(geo_res["edges"], cmap="gray")
    axes[1, 0].set_title(f"Canny Edges ({geo_res['line_count']} lines)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Segmented & Classified ({len(objects)} objs)")
    axes[1, 1].axis("off")

    plot_path = os.path.join(output_dir, "pipeline_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    annotated_path = os.path.join(output_dir, "annotated_result.png")
    cv2.imwrite(annotated_path, annotated)

    print("\n================ PIPELINE SUMMARY ================")
    print(f"Edges Processed     : Present")
    print(f"Corners Detected    : {geo_res['corner_count']}")
    print(f"Hough Lines Found   : {geo_res['line_count']}")
    print(f"Objects Detected    : {len(objects)}")
    print(f"Saved Diagnostic    : {plot_path}")
    print(f"Saved Annotated Img : {annotated_path}")
    print("==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionAudit: CLI Computer Vision Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--outdir", type=str, default="output", help="Directory to save artifacts")
    args = parser.parse_args()

    run_pipeline(args.image, args.outdir)