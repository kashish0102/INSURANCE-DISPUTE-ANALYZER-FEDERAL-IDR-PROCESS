# INSURANCE-DISPUTE-ANALYZER-FEDERAL-IDR-PROCESS
## Overview
This project is an AI-powered document processing system designed to analyze insurance dispute PDFs under the Federal IDR process.
It extracts objection or denial reasons, detects supporting documents, and retrieves structured metadata from both text and tables.

The system combines OCR, computer vision preprocessing, and LLM-based reasoning in an end-to-end pipeline with a Streamlit interface.

---

## Features

* Extracts text from PDFs using pdfplumber with OCR fallback
* Performs OCR using PaddleOCR and Tesseract for scanned documents
* Detects and reconstructs tables using OpenCV contour processing
* Uses Google Gemma (HuggingFace Transformers) to identify denial reasons
* Extracts structured identifiers such as TIN and NPI
* Detects supporting document filenames
* Provides optional voice output using gTTS
* Includes caching for faster repeated inference

---

## Tech Stack

Python
Streamlit
OpenCV
PaddleOCR
Tesseract OCR
HuggingFace Transformers (Gemma)
PyTorch
pdfplumber
Pandas

---

## How It Works

1. Upload an insurance PDF through the Streamlit interface
2. The system extracts text directly from the PDF
3. If text is missing, OCR is applied on page images
4. Tables are detected and reconstructed using image processing
5. Extracted text is passed to an LLM to identify rejection reasons
6. Structured data and supporting documents are returned to the user

---

## Notes

This project was developed as part of an industry training exercise focused on building real-world AI document processing workflows.
