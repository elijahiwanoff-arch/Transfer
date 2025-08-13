# pole_inspector_tool.py

import os
import sys
from collections import Counter

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Resolve path to the bundled model file (for PyInstaller one-file EXE)
# -----------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    base = sys._MEIPASS  # type: ignore[attr-defined]
else:
    base = os.path.dirname(__file__)

YOLO_MODEL_PATH = os.path.join(
    base, "runs", "detect", "pole_tag_detector12", "weights", "best.pt"
)

YOLO_CONF_THRESH = 0.5
CLARITY_THRESHOLD = 200.0  # higher → stricter clarity requirement

# Tune these for your camera / compression artifacts
PIXELATION_BLOCK_SIZE = 8  # tile width/height in pixels
PIXELATION_EDGE_THRESH = 30.0  # minimal variance at block boundaries
PIXELATION_PCT_BLOCKS = 0.5  # fraction of blocks that must look pixelated

# -----------------------------------------------------------------------------
# Initialize models
# -----------------------------------------------------------------------------
_yolo = YOLO(YOLO_MODEL_PATH)
_yolo.conf = YOLO_CONF_THRESH  # note: attribute exists in many Ultralytics builds

_reader = easyocr.Reader(["en"], gpu=False)

# -----------------------------------------------------------------------------
# Utility: measure image clarity (focus)
# -----------------------------------------------------------------------------
def measure_clarity(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# -----------------------------------------------------------------------------
# Utility: detect blocky pixelation artifacts
# -----------------------------------------------------------------------------
def is_pixelated(
    img: np.ndarray,
    block_size: int = PIXELATION_BLOCK_SIZE,
    edge_thresh: float = PIXELATION_EDGE_THRESH,
    pct_blocks: float = PIXELATION_PCT_BLOCKS,
) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blocks_x = w // block_size
    blocks_y = h // block_size
    pixelated_count = 0

    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0, x0 = by * block_size, bx * block_size
            block = gray[y0 : y0 + block_size, x0 : x0 + block_size]
            var_in = block.var()

            # check right-edge variance for block boundaries
            if bx < blocks_x - 1:
                edge = gray[y0 : y0 + block_size, x0 + block_size : x0 + block_size + 1]
                edge_var = edge.var()
                if var_in < 5.0 and edge_var > edge_thresh:
                    pixelated_count += 1

    total_blocks = blocks_x * blocks_y
    return (pixelated_count / total_blocks) >= pct_blocks


# -----------------------------------------------------------------------------
# Utility: combined digit-count matching (strict)
# -----------------------------------------------------------------------------
def combined_digit_count_match(expected: str, texts: list[str]) -> bool:
    """
    Return True if across all `texts` we see at least the same count of each digit
    in `expected` (ignoring order / spacing / extra chars), with no missing digits.
    """
    exp_ctr = Counter(ch for ch in expected if ch.isdigit())
    got_ctr = Counter()
    for txt in texts:
        got_ctr.update(ch for ch in txt if ch.isdigit())
    for digit, cnt in exp_ctr.items():
        if got_ctr.get(digit, 0) < cnt:
            return False
    return True


# -----------------------------------------------------------------------------
# Detection: YOLO pipeline
# -----------------------------------------------------------------------------
def detect_pole_tags_and_labels(img: np.ndarray):
    res = _yolo(img, verbose=False)[0]
    dets = res.boxes.xyxy.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy()

    tag_boxes, label_boxes = [], []
    annotated = img.copy()

    for (x1, y1, x2, y2), cls in zip(dets, classes):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        w, h = x2 - x1, y2 - y1
        color = (0, 0, 255) if int(cls) == 0 else (0, 255, 0)
        (tag_boxes if int(cls) == 0 else label_boxes).append([x1, y1, w, h])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    return tag_boxes, label_boxes, annotated


# -----------------------------------------------------------------------------
# OCR: read text from each label box
# -----------------------------------------------------------------------------
def ocr_from_label_boxes(
    img: np.ndarray,
    label_boxes: list[list[int]],
    min_conf: float = 0.15,
) -> list[str]:
    texts = []
    for x, y, w, h in label_boxes:
        crop = img[y : y + h, x : x + w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = _reader.readtext(rgb, detail=1, paragraph=False)
        lines = [
            txt
            for _, txt, conf in sorted(results, key=lambda t: -t[2])
            if conf >= min_conf
        ]
        texts.append("\n".join(lines))
    return texts


# -----------------------------------------------------------------------------
# Core: full pipeline + PASS/FAIL logic
# -----------------------------------------------------------------------------
def process_image(
    image_src: np.ndarray | str,
    filename: str | None = None,
    ocr_conf_threshold: float = 0.15,
) -> dict:
    # --- load image & filename ---
    if isinstance(image_src, str):
        img = cv2.imread(image_src)
        if img is None:
            raise FileNotFoundError(f"Cannot read {image_src}")
        filename = filename or os.path.basename(image_src)
    else:
        img = image_src.copy()
        filename = filename or "<uploaded>"

    # --- clarity check ---
    clarity_score = measure_clarity(img)
    clarity_pass = clarity_score >= CLARITY_THRESHOLD

    # --- pixelation check ---
    pixelation_fail = is_pixelated(img)
    pixelation_pass = not pixelation_fail

    # --- tag & label detection ---
    tag_boxes, label_boxes, annotated = detect_pole_tags_and_labels(img)

    # --- OCR crops ---
    label_texts = ocr_from_label_boxes(img, label_boxes, min_conf=ocr_conf_threshold)

    # --- combined digit match check ---
    pole_num = (filename or "").split("_", 1)[0]
    text_match_pass = combined_digit_count_match(pole_num, label_texts)

    # --- final PASS/FAIL ---
    status = "PASS" if (tag_boxes and label_boxes and text_match_pass) else "FAIL"

    return {
        "filename": filename,
        "clarity_score": clarity_score,
        "clarity_pass": clarity_pass,
        "pixelation_pass": pixelation_pass,
        "pixelation_fail": pixelation_fail,
        "tags_count": len(tag_boxes),
        "labels_count": len(label_boxes),
        "text_match_pass": text_match_pass,
        "status": status,
        "tag_boxes": tag_boxes,
        "label_boxes": label_boxes,
        "label_texts": label_texts,
        "annotated": annotated,
    }


# Alias for backward compatibility
process_image_path = process_image

# -----------------------------------------------------------------------------
# Optional: update your low-quality copy logic to include pixelation
# -----------------------------------------------------------------------------
# Example (in your main loop or GUI handler):
#
# if lowquality_enabled and (not result["clarity_pass"] or result["pixelation_fail"]):
#     shutil.copy2(
#         img_path,
#         os.path.join(out_base, "low_quality", os.path.basename(img_path)),
#     )
