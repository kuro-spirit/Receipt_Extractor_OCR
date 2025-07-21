import ollama
from paddleocr import PaddleOCR
import json
import os
import regex
import datetime

# --- Step 1: Initialize PaddleOCR ---
# Use use_textline_orientation instead of use_angle_cls for better compatibility
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# --- Step 2: Function to perform OCR ---
def perform_ocr(image_path):
    """
    Performs OCR on an image using PaddleOCR and returns a concatenated string of recognized text.
    Handles the dictionary-based result structure from PaddleOCR.
    """

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
    Output Json only:
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
            json_string = extract_json_block(llm_output)
            if json_string:
                extracted_data = json.loads(json_string)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"outputs/receipt_{timestamp}.json", "w", encoding="utf-8") as f:
                    json.dump(extracted_data, f, indent=4)
                return extracted_data
            else:
                raise ValueError("No JSON block found in LLM output.")
        except Exception as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"LLM Raw Output:\n{llm_output}")
            return {"error": "Could not parse LLM output as JSON.", "raw_llm_output": llm_output}

    except ollama.ResponseError as e:
        print(f"Error communicating with Ollama: {e}")
        print("Please ensure your Ollama server is running (`ollama serve`) and the specified model is pulled (`ollama pull llama2:7b`).")
        return {"error": "Ollama communication error."}
    
def extract_json_block(text):
    """
    Extracts the first JSON object from a string using a regular expression.
    This helps when the LLM adds extra explanation around the JSON output.
    """
    json_match = regex.search(r'\{(?:[^{}]|(?R))*\}', text, regex.DOTALL)
    if json_match:
        return json_match.group(0)
    return None

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