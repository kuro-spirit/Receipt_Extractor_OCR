import ollama
from paddleocr import PaddleOCR
import json
import os

# --- Step 1: Initialize PaddleOCR ---
# Use use_textline_orientation instead of use_angle_cls for better compatibility
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# --- Step 2: Function to perform OCR ---
def perform_ocr(image_path):
    """
    Performs OCR on an image using PaddleOCR and returns a concatenated string of recognized text.
    Handles the dictionary-based result structure from PaddleOCR.
    """
    try:
        # Perform OCR. The result is typically a list, where each element represents a page.
        # For a single-page receipt, result[0] will contain the OCR details for that page.
        # This result[0] is a dictionary with keys like 'rec_texts', 'rec_scores', etc.
        result = ocr.ocr(image_path)
        extracted_text = ""

        # Check if result is not empty and the first page's result is a dictionary
        if result and isinstance(result[0], dict):
            page_results = result[0]
            
            # Ensure 'rec_texts' key exists and it's a list
            if 'rec_texts' in page_results and isinstance(page_results['rec_texts'], list):
                for text in page_results['rec_texts']:
                    if isinstance(text, str): # Ensure the item is a string
                        extracted_text += text + "\n"
            else:
                print("Warning: 'rec_texts' key not found or not a list in OCR result for the first page.")
        else:
            print("Warning: OCR did not return a valid dictionary for the first page or no result found.")
        
        return extracted_text.strip() # Remove leading/trailing whitespace

    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None

# --- Step 3: Function to extract info using Ollama LLM ---
def extract_info_with_llm(ocr_text):
    """
    Sends OCR text to Ollama LLM for structured information extraction.
    """
    if not ocr_text:
        return {"error": "No text provided for LLM extraction."}

    prompt = f"""
    You are an intelligent assistant specialized in extracting information from receipt text.
    Analyze the following receipt text and extract the 'Date' (in YYYY-MM-DD format), 'Description' (a list of items purchased with their individual amounts), and 'Total_Amount'.
    If a specific item description is not clear, use a general description like "Various Groceries".
    If the date or amount is missing or unclear, indicate "N/A".
    Provide the output in a JSON format.

    Receipt Text:
    ---
    {ocr_text}
    ---

    JSON Output Example:
    {{
        "Date": "YYYY-MM-DD",
        "Description": [
            {{"item": "Coffee", "amount": 4.50}},
            {{"item": "Sandwich", "amount": 8.99}}
        ],
        "Total_Amount": 13.49
    }}
    """

    try:
        # Use Ollama's chat function. Changed model to 'llama2:7b' based on your 'ollama list' output.
        # Ensure your Ollama server is running (`ollama serve`) and the model is pulled (`ollama pull llama2:7b`).
        response = ollama.chat(
            model='llama2:7b', # This line has been updated
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.1 # Lower temperature for more deterministic output
            }
        )
        llm_output = response['message']['content']

        # Attempt to parse the LLM's response as JSON
        try:
            # LLMs sometimes wrap JSON in markdown, so we need to clean it
            if '```json' in llm_output:
                llm_output = llm_output.split('```json')[1].split('```')[0].strip()
            elif '```' in llm_output: # Sometimes just triple backticks without 'json'
                llm_output = llm_output.split('```')[1].split('```')[0].strip()

            extracted_data = json.loads(llm_output)
            return extracted_data
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"LLM Raw Output:\n{llm_output}")
            return {"error": "Could not parse LLM output as JSON.", "raw_llm_output": llm_output}

    except ollama.ResponseError as e:
        print(f"Error communicating with Ollama: {e}")
        print("Please ensure your Ollama server is running (`ollama serve`) and the specified model is pulled (`ollama pull llama2:7b`).")
        return {"error": "Ollama communication error."}
    except Exception as e:
        print(f"An unexpected error occurred during LLM extraction: {e}")
        return {"error": "Unexpected LLM extraction error."}

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure this path points to your resized receipt image for better performance
    receipt_image_path = 'test_receipt_small.png' # Or 'sample_receipt_small.jpg'
    
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

            print("\n--- Extracted Structured Data ---")
            print(json.dumps(extracted_data, indent=4))
        else:
            print("OCR failed or returned no text. Check image quality and OCR configuration.")