# Receipt Extractor OCR
This project extracts structured data from receipt images using PaddleOCR and LLM-based parsing.
It processes scanned/photographed receipts and returns:
- Items purchased + price
- Total price

## Features
Preprocessing for noisy or skewed images
OCR with layout-aware text extraction
LLM-based parsing of text into JSON

## Installation
git clone https://github.com/kuro-spirit/Receipt_Extractor_OCR.git  
cd Receipt_Extractor_OCR  
python -m venv venv  
.\venv\Scripts\activate  
pip install -r requirements.txt  

## Structure
.  
├── ImagePreprocessing.py       # Resizing, thresholding, denoising, etc.  
├── llmExtraction.py            # Prompts and parsing logic  
├── Receipt_Extractor.py        # Orchestrates the full pipeline  
├── receipts/                   # Receipt storage  
├── ouputs/                     # Output Json  
├── requirements.txt  
└── README.md  

## Pipeline
Receipt Image ➝ Preprocessing ➝ OCR ➝ Text ➝ LLM Prompt ➝ JSON Output
