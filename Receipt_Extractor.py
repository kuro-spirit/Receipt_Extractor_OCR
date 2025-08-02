import os
from paddleocr import PaddleOCR
from ImagePreprocessing import preprocess_image
from llmExtraction import extract_info_with_llm
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv
from inference import get_model

model = get_model('receipts-nrlrs/2')
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.5)

def perform_layout_ocr(image_path, conf_thresh=0.25):
    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 3. Run YOLOv8 to detect objects
    results = model.predict(image, conf=conf_thresh)

    # Collect cropped text regions
    all_text = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls)
            label = r.names[cls_id]  # class name
            if label not in ["person", "car"]:  # filter only likely text regions
                # xyxy box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = image[y1:y2, x1:x2]
                # OCR on cropped region
                ocr_result = ocr.predict(cropped)
                if ocr_result and len(ocr_result[0]) > 0:
                    for line in ocr_result[0]:
                        all_text.append(line[1][0])

    # Combine all recognized text lines
    return "\n".join(all_text)

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure this path points to your resized receipt image for better performance
    receipt_image_path = 'Receipts/test_receipt_small.png' # Or 'sample_receipt_small.jpg'
    
    if not os.path.exists(receipt_image_path):
        print(f"Error: Receipt image not found at '{receipt_image_path}'. Please ensure the image exists and the path is correct.")
    else:
        print(f"--- Processing receipt: {receipt_image_path} ---")

        # 1. Perform OCR to get raw text
        print("Performing OCR...")
        ocr_text_output = perform_layout_ocr(receipt_image_path)

        if ocr_text_output:
            print("\n--- OCR Text Output ---")
            print(ocr_text_output)

            # 2. Extract information using Ollama LLM
            print("\n--- Extracting information with Ollama LLM ---")
            extracted_data = extract_info_with_llm(ocr_text_output)
            print(extracted_data)

        else:
            print("OCR failed or returned no text. Check image quality and OCR configuration.")