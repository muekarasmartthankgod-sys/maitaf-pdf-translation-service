import os
import json
import streamlit as st
import pymupdf
from openai import OpenAI

st.set_page_config(page_title="MAITAF Customs AI Lab", layout="centered")
st.title("📄 Customs PDF Translator Lab")
st.subheader("High-Fidelity Grid Alignment Translation Node")

# --- UI CONTROLS SELECTION AT TOP ---
st.markdown("### 🌐 Step 1: Configure Your Trade Corridor")
col1, col2 = st.columns(2)

supported_languages = ["English", "French", "Spanish", "German", "Mandarin Chinese", "Arabic", "Portuguese", "Italian", "Dutch"]

with col1:
    source_lang = st.selectbox("Translate From (Source Language) :", ["Auto-Detect"] + supported_languages)
with col2:
    target_lang = st.selectbox("Translate To (Target Destination) :", supported_languages)

# --- CREDENTIAL SECURE VALIDATION LAYER ---
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if api_key:
    api_key = api_key.strip().strip('"').strip("'")

if not api_key:
    st.error("❌ GROQ_API_KEY is missing in your App Secrets!")
else:
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def translate_structured_payload(text_lines_dict: dict, src: str, tgt: str) -> dict:
        if not text_lines_dict:
            return {}
        
        prompt_payload = (
            f"You are an expert multilingual international trade, logistics, and customs compliance translation engine.\n"
            f"Your task is to translate the values inside this JSON object from {src} into {tgt}.\n\n"
            "STRICT COMPLIANCE DIRECTIVES:\n"
            f"1. Translate all descriptive phrases, table headers, and legal disclaimers accurately into {tgt}.\n"
            "2. Keep all numbers, internal codes, IDs, quantities, prices, dates, bank IBANs, and currency symbols EXACTLY as they are.\n"
            "3. Preserve all standard trade abbreviations (HS Codes, Incoterms, VAT, TTC, HT).\n"
            "4. Return ONLY a valid JSON object matching the input keys perfectly. Do not add introductory or explanatory text.\n\n"
            f"Target Objective JSON: {json.dumps(text_lines_dict, ensure_ascii=False)}"
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
            return text_lines_dict

    st.markdown("### 📁 Step 2: Upload Trade Documentation")
    uploaded_file = st.file_uploader("Drop invoice, packing list, or manifest here", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Run High-Fidelity Grid Translation ➔", use_container_width=True):
            with st.spinner("Processing document grid alignment layers..."):
                try:
                    input_bytes = uploaded_file.read()
                    doc = pymupdf.open(stream=input_bytes, filetype="pdf")
                    
                    for page in doc:
                        words = page.get_text("words")
                        
                        # Sort words vertically into precise rows
                        lines_map = {}
                        for w in words:
                            x0, y0, x1, y1, text, block_no, line_no, word_no = w
                            line_key = int(y1)
                            
                            found_key = None
                            for existing_key in lines_map.keys():
                                if abs(existing_key - line_key) <= 3:
                                    found_key = existing_key
                                    break
                                    
                            if found_key is not None:
                                lines_map[found_key].append(w)
                            else:
                                lines_map[line_key] = [w]
                        
                        # Sort words horizontally within each row
                        sorted_lines_payload = {}
                        line_index = 0
                        for k in sorted(lines_map.keys()):
                            row_words = sorted(lines_map[k], key=lambda x: x[0])
                            full_line_string = " ".join([word[4] for word in row_words])
                            
                            if full_line_string.strip():
                                min_x = min([word[0] for word in row_words])
                                max_x = max([word[2] for word in row_words])
                                min_y = min([word[1] for word in row_words])
                                max_y = max([word[3] for word in row_words])
                                
                                sorted_lines_payload[f"line_{line_index}"] = {
                                    "text": full_line_string,
                                    "coords": (min_x, min_y, max_x, max_y)
                                }
                                line_index += 1
                        
                        # Translate batch map payload
                        translation_input = {key: val["text"] for key, val in sorted_lines_payload.items()}
                        translated_output = translate_structured_payload(translation_input, source_lang, target_lang)
                        
                        # Redact and re-write high precision rows
                        for key, val in sorted_lines_payload.items():
                            x0, y0, x1, y1 = val["coords"]
                            t_text = translated_output.get(key, val["text"])
                            
                            page.add_redact_annot(pymupdf.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=(1, 1, 1))
                            page.apply_redactions()
                            
                            page.insert_text(pymupdf.Point(x0, y1), t_text, fontsize=8, fontname="helv", color=(0, 0, 0))
                            
                    output_bytes = doc.tobytes()
                    doc.close()
                    st.success("✓ Translation engine grid reconstruction complete!")
                    st.download_button(
                        label="Download Perfect Aligned PDF 📥",
                        data=output_bytes,
                        file_name=f"fixed_grid_{target_lang}_{uploaded_file.name}",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as ex:
                    st.error(f"Fatal Engine Runtime Failure: {ex}")
