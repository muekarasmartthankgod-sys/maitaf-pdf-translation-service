import os
import json
import streamlit as st
import pymupdf
import pdfplumber
from openai import OpenAI

st.set_page_config(page_title="MAITAF Customs AI Lab", layout="centered")
st.title("📄 Customs PDF Translator Lab")
st.subheader("High-Fidelity Table Extraction Node")

st.markdown("### 🌐 Step 1: Configure Your Trade Corridor")
col1, col2 = st.columns(2)

supported_languages = ["English", "French", "Spanish", "German", "Mandarin Chinese", "Arabic", "Portuguese", "Italian", "Dutch"]

with col1:
    source_lang = st.selectbox("Translate From (Source Language) :", ["Auto-Detect"] + supported_languages)
with col2:
    target_lang = st.selectbox("Translate To (Target Destination) :", supported_languages)

api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if api_key:
    api_key = api_key.strip().strip('"').strip("'")

if not api_key:
    st.error("❌ GROQ_API_KEY is missing in your App Secrets!")
else:
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def translate_payload_dictionary(payload: dict, src: str, tgt: str) -> dict:
        if not payload:
            return {}
        
        prompt_payload = (
            f"You are an expert multilingual international trade, logistics, and customs compliance translation engine.\n"
            f"Translate the text values inside this JSON object from {src} into {tgt}.\n\n"
            "STRICT MANDATES:\n"
            f"1. Translate all industry descriptions, legal disclaimers, and headers into {tgt}.\n"
            "2. Keep all numbers, metrics, quantities, prices, dates, currency symbols, and bank codes EXACTLY as they are.\n"
            "3. Maintain standard logistics shorthand (such as HS Codes, Incoterms).\n"
            "4. Return ONLY a valid JSON object matching the exact input keys perfectly.\n\n"
            f"Target Objective JSON: {json.dumps(payload, ensure_ascii=False)}"
        )
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt_payload}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"Groq API Processing Failure: {e}")
            return payload

    st.markdown("### 📁 Step 2: Upload Trade Documentation")
    uploaded_file = st.file_uploader("Drop invoice, packing list, or manifest here", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Run High-Fidelity Table Grid Translation ➔", use_container_width=True):
            with st.spinner("Processing structural table cells..."):
                try:
                    input_bytes = uploaded_file.read()
                    pymupdf_doc = pymupdf.open(stream=input_bytes, filetype="pdf")
                    
                    with pdfplumber.open(stream=input_bytes) as plumber_doc:
                        for page_idx, plumber_page in enumerate(plumber_doc):
                            pymupdf_page = pymupdf_doc[page_idx]
                            
                            tables = plumber_page.find_tables()
                            table_areas = []
                            translation_manifest = {}
                            element_index = 0
                            
                            for table in tables:
                                table_areas.append(table.bbox)
                                table_data = table.extract()
                                for row in table_data:
                                    for cell_text in row:
                                        if cell_text and cell_text.strip():
                                            if not cell_text.strip().replace(",", "", 1).replace(".", "", 1).replace("€", "").strip().isdigit():
                                                key = f"table_{element_index}"
                                                translation_manifest[key] = cell_text.strip()
                                                element_index += 1
                            
                            blocks = pymupdf_page.get_text("blocks")
                            loose_blocks = []
                            for b in blocks:
                                bx0, by0, bx1, by1, b_text, b_no, b_type = b[:7]
                                if b_text.strip():
                                    in_table = False
                                    for t_bbox in table_areas:
                                        tx0, ty0, tx1, ty1 = t_bbox
                                        if not (bx1 < tx0 or bx0 > tx1 or by1 < ty0 or by0 > ty1):
                                            in_table = True
                                            break
                                    if not in_table and not b_text.strip().replace(".", "", 1).isdigit():
                                        key = f"loose_{element_index}"
                                        translation_manifest[key] = b_text.strip()
                                        loose_blocks.append((key, b))
                                        element_index += 1
                            
                            translated_data = translate_payload_dictionary(translation_manifest, source_lang, target_lang)
                            
                            for key, b_info in loose_blocks:
                                bx0, by0, bx1, by1, b_text, b_no, b_type = b_info[:7]
                                t_text = translated_data.get(key, b_text)
                                pymupdf_page.add_redact_annot(pymupdf.Rect(bx0, by0, bx1, by1), fill=(1, 1, 1))
                                pymupdf_page.apply_redactions()
                                pymupdf_page.insert_textbox(pymupdf.Rect(bx0, by0, bx1, by1 + 10), t_text, fontsize=8, fontname="helv", color=(0, 0, 0))
                            
                            cell_element_idx = 0
                            for table in tables:
                                table_data = table.extract()
                                for row_idx, row in enumerate(table_data):
                                    for col_idx, cell_text in enumerate(row):
                                        if cell_text and cell_text.strip():
                                            key = f"table_{cell_element_idx}"
                                            t_text = translated_data.get(key, cell_text.strip()) if key in translation_manifest else cell_text
                                            cell_obj = table.cells[row_idx][col_idx]
                                            if cell_obj:
                                                cx0, cy0, cx1, cy1 = cell_obj
                                                pymupdf_page.add_redact_annot(pymupdf.Rect(cx0 + 2, cy0 + 2, cx1 - 2, cy1 - 2), fill=(1, 1, 1))
                                                pymupdf_page.apply_redactions()
                                                pymupdf_page.insert_textbox(pymupdf.Rect(cx0 + 3, cy0 + 3, cx1 - 3, cy1), t_text, fontsize=7.5, fontname="helv", color=(0, 0, 0))
                                            cell_element_idx += 1
                                            
                    output_bytes = pymupdf_doc.tobytes()
                    pymupdf_doc.close()
                    st.success("✓ Table structure processing complete!")
                    st.download_button(
                        label="Download Perfect Aligned PDF 📥",
                        data=output_bytes,
                        file_name=f"grid_fixed_{uploaded_file.name}",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as ex:
                    st.error(f"Fatal Engine Runtime Failure: {ex}")
