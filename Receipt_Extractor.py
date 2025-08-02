import os
from paddleocr import PaddleOCR
from ImagePreprocessing import preprocess_image
from llmExtraction import extract_info_with_llm
import numpy as np

import layoutparser as lp
from PIL import Image


# Use use_textline_orientation instead of use_angle_cls for better compatibility
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.5)

model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config.yaml",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

def perform_ocr(image_path):
    """
    Performs OCR on an image using PaddleOCR and returns a concatenated string of recognized text.
    Handles the dictionary-based result structure from PaddleOCR.
    """
    preprocessed_path = preprocess_image(image_path)

    image = Image.open(preprocessed_path)
    layout = model.detect(image)
    text_blocks = [b for b in layout if b.type in ["Text", "Title", "List"]]

    for i, block in enumerate(text_blocks):
        segment = (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2)
        cropped = image.crop(segment)
        cropped.save(f"outputs/block_{i}.png")

    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    full_text = []
    for block in text_blocks:
        (x1, y1, x2, y2) = (block.block.x_1, block.block.y_1,
                            block.block.x_2, block.block.y_2)
        cropped = image.crop((x1, y1, x2, y2))
        results = ocr.predict(np.array(cropped), cls=True)
        for line in results[0]:
            full_text.append(line[1][0])  # Extract text

    # result = ocr.predict(preprocessed_path)
    extracted_text = "\n".join(full_text)

    # Check if result is not empty and the first page's result is a dictionary
    # if result and isinstance(result[0], dict):
    #     page_results = result[0]
        
    #     # Ensure 'rec_texts' key exists and it's a list
    #     if 'rec_texts' in page_results and isinstance(page_results['rec_texts'], list):
    #         for text in page_results['rec_texts']:
    #             if isinstance(text, str): # Ensure the item is a string
    #                 extracted_text += text + "\n"
    
    return extracted_text.strip() # Remove leading/trailing whitespace

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
        ocr_text_output = perform_ocr(receipt_image_path)

        if ocr_text_output:
            print("\n--- OCR Text Output ---")
            print(ocr_text_output)

            # 2. Extract information using Ollama LLM
            print("\n--- Extracting information with Ollama LLM ---")
            extracted_data = extract_info_with_llm(ocr_text_output)
            print(extracted_data)

        else:
            print("OCR failed or returned no text. Check image quality and OCR configuration.")