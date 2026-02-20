import streamlit as st
from transformers import AutoProcessor, GemmaForCausalLM
import torch
import pdfplumber
import pytesseract
import re
from PIL import Image
import io
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import pandas as pd
from paddleocr import PaddleOCR
import csv
import matplotlib.pyplot as plt
from difflib import get_close_matches
import os
import pickle
import hashlib
from difflib import get_close_matches

ocr = PaddleOCR(use_angle_cls=True, lang='en') 
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(key):
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.pkl")

@st.cache_resource
def load_gemma_model():
    model_id = "google/gemma-2b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = GemmaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )
    return processor, model

processor, model = load_gemma_model()

def extract_or_fallback(text):
    lines = text.splitlines()
    extracting = False
    extracted_section = []

    for line in lines:
        if "Federal IDR Process Applicability Attestation" in line:
            extracting = True
            continue
        if extracting:
            extracted_section.append(line)
            if "sign & submit" in line.lower():
                break

    return "\n".join(extracted_section).strip() if extracted_section else text.strip()

def extract_text_from_pdf(pdf_file):
    full_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return extract_or_fallback(full_text)

def correct_image_orientation(image):
    try:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        osd = pytesseract.image_to_osd(image_cv)
        rotation = 0
        for line in osd.split('\n'):
            if 'Rotate:' in line:
                rotation = int(line.split(':')[-1].strip())
                break
        if rotation != 0:
            image = image.rotate(-rotation, expand=True)
    except Exception as e:
        print("Orientation correction failed:", e)
    return image
    

def extract_text_and_tables_combined(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    img_bin = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    kernel_len = gray.shape[1] // 100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    vertical_lines = cv2.dilate(cv2.erode(img_bin, ver_kernel, iterations=1), ver_kernel, iterations=2)
    horizontal_lines = cv2.dilate(cv2.erode(img_bin, hor_kernel, iterations=1), hor_kernel, iterations=2)

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    contours, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][1]))

    box = []
    for x, y, w, h in boundingBoxes:
        if 10 < w < 1000 and 10 < h < 500:
            box.append([x, y, w, h])
    box = sorted(box, key=lambda b: (b[1], b[0]))

    rows = []
    current_row = []
    mean_height = np.mean([b[3] for b in box]) if box else 20
    tolerance = mean_height * 0.6
    last_y = -1

    for b in box:
        x, y, w, h = b
        if last_y == -1 or abs(y - last_y) <= tolerance:
            current_row.append(b)
            last_y = (last_y + y) / 2 if last_y != -1 else y
        else:
            rows.append(sorted(current_row, key=lambda r: r[0]))
            current_row = [b]
            last_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda r: r[0]))

    table_data = []
    for row_cells in rows:
        row_text = []
        for x, y, w, h in row_cells:
            cropped = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(cropped, config='--psm 6')
            clean_text = ' '.join(text.strip().splitlines())
            row_text.append(clean_text if clean_text else "NA")
        table_data.append(row_text)

    if len(table_data) >= 2 and len(table_data[0]) == len(table_data[1]):
        row0, row1 = table_data[0], table_data[1]
        merged_header = [f"{r0} {r1}".strip() for r0, r1 in zip(row0, row1)]
        table_data = [merged_header] + table_data[2:]

    max_len = max(len(r) for r in table_data) if table_data else 0
    for row in table_data:
        row.extend(["NA"] * (max_len - len(row)))

    def clean_and_make_unique(header):
        seen = {}
        new_header = []
        for i, col in enumerate(header):
            col = col.strip() if col else f"Column_{i}"
            col = " ".join(col.split())
            if not col or col.upper() == "NA":
                col = f"Column_{i}"
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 1
            new_header.append(col)
        return new_header

    table_dfs = []
    current_table = []
    for row in table_data:
        if any(keyword in cell.upper() for cell in row for keyword in ["DATE", "AMOUNT", "SERVICE", "CHARGE", "PAYMENT"]) and current_table:
            header = clean_and_make_unique(current_table[0])
            data = current_table[1:]
            df = pd.DataFrame(data, columns=header)
            if len(df) >= 3: 
                table_dfs.append(df)
            current_table = []
        current_table.append(row)

    if current_table:
        header = clean_and_make_unique(current_table[0])
        data = current_table[1:]
        df = pd.DataFrame(data, columns=header)
        if len(df) >= 3: 
            table_dfs.append(df)

    plain_text = pytesseract.image_to_string(image)

    return plain_text.strip(), table_dfs




def ocr_from_pdf(pdf_file):
  
    table_list = []
    full_text = ""

    with pdfplumber.open(pdf_file) as pdf:
         for page_num, page in enumerate(pdf.pages):
            img = page.to_image(resolution=300).original
            img = correct_image_orientation(img)
            img = img.convert("RGB")
            # st.markdown(f"Page {page_num + 1}")
            # st.image(img, caption="Preprocessed Image Before OCR", use_column_width=True)

            text, table_dfs = extract_text_and_tables_combined(img)
            for df in table_dfs:
                table_str = df.to_string(index=False)
                combined_text = text + "\n\n--- Extracted Table ---\n" + table_str
                full_text += combined_text + "\n\n"
                table_list.append(df)

    return extract_or_fallback(full_text), table_list


def find_supporting_documents(text):
    pattern = r'\b(?:[A-Za-z0-9_\-()\[\] ]{1,50}\s*\.pdf|\S+\.txt|\S+\.docx|\S+\.xlsx|\S+\.csv)\b'

    matches = re.findall(pattern, text)
    cleaned_matches = []

    for match in matches:
        filename = match.strip()

        if filename.endswith(" "):
            cleaned_matches.append(f"{filename}[REDACTED].pdf")
        else:
            cleaned_matches.append(f"{filename}")

    return cleaned_matches

@st.cache_data
def gemma_infer_rejection_reason(text):
    cache_path = get_cache_path(text[:500]) 

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    prompt = (
        "Extract all objection or denial reason(s) from the extracted text from the document. "
        "Focus on why the dispute is not eligible under the Federal IDR Process or was denied for payment.\n\n"
        "Only include complete sentences that directly state an objection or denial reason. "
        "Exclude any supporting explanations, elaborations, or paraphrased versions of the same reason.\n\n"
        "Examples of valid objections: dispute submitted early, eligible for state process, prior to applicable policy year, plan not subject to NSA, "
        "item or service not covered by plan, item or service not NSA eligible, exceeded four-day timeline, incorrectly batched, "
        "incorrectly bundled, notice of initiation not submitted, open negotiation not complete, open negotiation not initiated, "
        "cooling off period not completed, IDR submitted early, dispute initiated before IDR window, we would like to have a staff member evaluate the disputes where the objection type falls into this category.  As we identify trends, we can provide an update to this list with action steps and the need for outreach as applicable, etc.\n\n"
        "Return only the objection or denial sentence(s) exactly as they appear in the document. "
        "Do not summarize or rephrase. Multiple objections may exist in one document.\n\n"
        f"Document Section:\n{text[:2000]}"
)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample= False,
        )
        generation = generation[0][input_len:]

    result = processor.decode(generation, skip_special_tokens=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result.strip()


def extract_columns_and_ids_from_tables(full_text, table_dfs, target_headers):
    tin_match = re.search(r'TIN[:\s]+(\d+)', full_text)
    npi_match = re.search(r'NPI[:\s]+(\d+)', full_text)

    tin = tin_match.group(1) if tin_match else "TIN not found"
    npi = npi_match.group(1) if npi_match else "NPI not found"

    all_matching_rows = []

    for df in table_dfs:
        df.columns = [col.strip().upper() for col in df.columns]
        simplified_columns = {re.sub(r'\W+', '', col.upper()): col for col in df.columns}
        col_map = {}

        for header in target_headers:
            simplified_target = re.sub(r'\W+', '', header.upper())
            match = get_close_matches(simplified_target, simplified_columns.keys(), n=1, cutoff=0.4)
            if match:
                col_map[header] = simplified_columns[match[0]]

        # Extract only if all requested columns are found in this df
        if len(col_map) == len(target_headers):
            for _, row in df.iterrows():
                row_data = {}
                skip_row = False
                for header in target_headers:
                    val = str(row[col_map[header]]).strip()
                    # Optional: you can keep the "LINE CTRL" check or remove it as needed
                    if header == "LINE CTRL" and re.search(r'[A-Za-z]', val):
                        skip_row = True
                        break
                    row_data[header] = val
                if not skip_row:
                    all_matching_rows.append(row_data)

    return tin, npi, pd.DataFrame(all_matching_rows)




def fix_mid_sentence_caps(text):
    words = text.split()
    fixed_words = [words[0]]  

    for prev, curr in zip(words, words[1:]):
        if curr[0].isupper() and not prev.endswith(('.', '!', '?')) and not curr.isupper():
           
            fixed_words.append(curr.lower())
        else:
            fixed_words.append(curr)

    return ' '.join(fixed_words)


def humanize_text(text):
    lines = text.strip().splitlines()
    sentence = ''
    for line in lines:
        sentence += ' ' + line.strip()

    text = sentence.strip()

    text = re.sub(r'\breason\(s\)', 'reasons', text, flags=re.IGNORECASE)

    text = re.sub(r'\b(however|therefore|in conclusion)\b', r'\1,', text, flags=re.IGNORECASE)

    text = re.sub(r'\s+([.,!?])', r'\1', text)

    text = re.sub(r'\s{2,}', ' ', text)

    acronym_phrases = [
        "Federal IDR", "Claim ID", "TIN Number", "NPI Number", "business day negotiation"
    ]
    for phrase in acronym_phrases:
        text = text.replace(phrase, phrase.replace(" ", "\u00A0"))

    if not text.endswith(('.', '!', '?')):
        text += '.'

    text = fix_mid_sentence_caps(text)

    return text.strip()

# Streamlit UI
st.title("Objection Extractor")
mode = st.selectbox("Select Upload Mode", ["Upload Full PDF", "Upload Supporting Documents"])
if mode == "Upload Full PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        extracted_text = extract_text_from_pdf(uploaded_file)
        extracted_text = re.sub(r'\s+', ' ',extracted_text)  
        if not extracted_text:
            st.warning(" Running OCR...")
            extracted_text = ocr_from_pdf(uploaded_file)
            extracted_text = re.sub(r'\s+', ' ', extracted_text)
            for i, df in enumerate(table_list):
                st.write(f"Extracted Table from Page")
                st.dataframe(df)

        st.text_area("Extracted Text", extracted_text, height=300)
        if st.button("Find Rejection Reason"):
        #  with st.spinner("Analyzing..."):

            extracted_text_for_model = extract_or_fallback(extracted_text)

            st.info("Please wait")
            reason = gemma_infer_rejection_reason(extracted_text_for_model)
            st.success(f"Objection: {reason}")
            tts = gTTS(humanize_text(reason), lang='en', slow=False)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_filename = fp.name
                tts.save(temp_filename)

            with open(temp_filename, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
        

            supporting_documents = find_supporting_documents(extracted_text_for_model)
            if supporting_documents:
                st.markdown("Supporting Documents Found:")
                for doc in supporting_documents:
                    st.markdown(f"- `{doc}`")
            else:
                st.markdown("No supporting documents found.")

elif mode == "Upload Supporting Documents":
    st.info("Upload your supporting documents (e.g., .pdf, .txt, .docx, etc.)")
    supporting_file = st.file_uploader("Upload a supporting document", type=["pdf", "txt", "docx"])
    
    if supporting_file:
        extracted_text = extract_text_from_pdf(supporting_file)
        extracted_text = re.sub(r'\s+', ' ', extracted_text)

        table_list = []  # Default empty table list

        if not extracted_text.strip():
            st.warning("Running OCR...")
            extracted_text, table_list = ocr_from_pdf(supporting_file)
            extracted_text = re.sub(r'\s+', ' ', extracted_text)

        st.text_area("Extracted Text", extracted_text, height=300)
        
        user_columns_input = st.text_input("Enter column names you want to extract (separated by commas):")
        user_columns = [col.strip().upper() for col in user_columns_input.split(",") if col.strip()] if user_columns_input else []

        if table_list:
            for i, df in enumerate(table_list):
                st.write(f"Extracted Table ")
                st.dataframe(df)

            tin, npi, extracted = extract_columns_and_ids_from_tables(extracted_text, table_list, user_columns)

            st.subheader("Extracted TIN and NPI")
            st.write(f"TIN: {tin}")
            st.write(f"NPI: {npi}")

            st.subheader("Extracted Table Columns")
            extracted_df = pd.DataFrame(extracted)
            st.dataframe(extracted_df)
        else:
            st.warning("No matching rows found with required columns.")
