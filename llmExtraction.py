import ollama
import json
import regex
import datetime

def extract_info_with_llm(ocr_text):
    """
    Sends OCR text to Ollama LLM for structured information extraction.
    """
    if not ocr_text:
        return {"error": "No text provided for LLM extraction."}

    prompt = f"""
    You are an intelligent assistant specialized in extracting structured information from raw OCR receipt text.

    Here is the receipt text:
    ---
    {ocr_text}
    ---

    Your response must be ONLY valid JSON and nothing else. Do not explain anything or include comments.

    Return a JSON object with the following format:
    {{
    "Date": "YYYY-MM-DD",                 // Receipt date
    "Description": [                      // List of items purchased
        {{
        "item": "<item name>",            // Cleaned name of the item
        "amount": <amount>                // Price as a float
        }}
    ],
    "Total_Amount": <total_price>         // Final total paid
    }}

    Notes:
    - Only include actual purchased items in the "Description". Do not include "Subtotal", "Total", "GST", etc.
    - Group related words like "Large Meat Supreme" into one item if appropriate.
    - Use your best judgment to clean up OCR noise (e.g. '@ $10.90' should just be 10.90).
    - The amount should be a float without a dollar sign.
    - Do not repeat the subtotal, tax, or total as items.
    - Output only a single valid JSON object and nothing else.
    - If the date or amount is missing or unclear, indicate "N/A".
    - If a specific item description is not clear, use a general description like "Various Groceries".
    - If multiple lines describe one item, combine them into a single entry. Only use the final listed price shown for the item. Do not split prices or infer sub-prices.
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