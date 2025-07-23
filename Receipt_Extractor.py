import os
from paddleocr import PaddleOCR
from ImagePreprocessing import preprocess_image
from llmExtraction import extract_info_with_llm

# Use use_textline_orientation instead of use_angle_cls for better compatibility
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.5)

def perform_ocr(image_path):
    """
    Performs OCR on an image using PaddleOCR and returns a concatenated string of recognized text.
    Handles the dictionary-based result structure from PaddleOCR.
    """
    preprocessed_path = preprocess_image(image_path)

    # Perform OCR. The result is typically a list, where each element represents a page.
    # For a single-page receipt, result[0] will contain the OCR details for that page.
    # This result[0] is a dictionary with keys like 'rec_texts', 'rec_scores', etc.
    result = ocr.predict(image_path)
    extracted_text = ""

    # Check if result is not empty and the first page's result is a dictionary
    if result and isinstance(result[0], dict):
        page_results = result[0]
        
        # Ensure 'rec_texts' key exists and it's a list
        if 'rec_texts' in page_results and isinstance(page_results['rec_texts'], list):
            for text in page_results['rec_texts']:
                if isinstance(text, str): # Ensure the item is a string
                    extracted_text += text + "\n"
    
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