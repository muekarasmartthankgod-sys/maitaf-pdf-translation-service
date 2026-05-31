import os
import json
import streamlit as st
import pymupdf
from openai import OpenAI

# Page configurations for a professional, clean user layout
st.set_page_config(page_title="MAITAF Customs AI Lab", layout="centered")
st.title("📄 Customs PDF Translator Lab")
st.subheader("Universal Trading Corridor Configuration Gateway")

# --- FIXED: FRONTEND DROPDOWN SELECTION PANEL ALWAYS AT THE TOP ---
st.markdown("### 🌐 Step 1: Configure Your Trade Corridor")
col1, col2 = st.columns(2)

supported_languages = [
    "English", "French", "Spanish", "German", "Mandarin Chinese", 
    "Arabic", "Portuguese", "Italian", "Dutch", "Japanese", 
    "Hindi", "Russian", "Korean", "Turkish", "Indonesian"
]

with col1:
    source_lang = st.selectbox("Translate From (Source Language) :", ["Auto-Detect"] + supported_languages)

with col2:
    target_lang = st.selectbox("Translate To (Target Destination) :", supported_languages)

# --- SECURE CREDENTIAL ROUTING LAYER ---
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if api_key:
    api_key = api_key.strip().strip('"').strip("'")

if not api_key:
    st.error("❌ GROQ_API_KEY is missing in your App Secrets!")
else:
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def translate_page_blocks(blocks_list: list, src: str, tgt: str) -> list:
        if not blocks_list:
            return []
        input_manifest = {f"block_{i}": text.strip() for i, text in enumerate(blocks_list)}
        
        prompt_payload = (
            f"You are an expert multilingual international trade and customs compliance translator.\n"
            f"Translate the values in this JSON object FROM {src} TO {tgt}.\n\n"
            "CRITICAL MANDATES:\n"
            f"1. Translate all industry and legal vocabulary accurately into {tgt}.\n"
            "2. Keep all numbers, metrics, quantities, prices, and symbols EXACTLY as they are.\n"
            "3. Retain standard global logistics abbreviations (HS Codes, Incoterms).\n"
            "4. Return ONLY a valid JSON object matching the structure.\n\n"
            f"Input Target Map: {json.dumps(input_manifest, ensure_ascii=False)}"
        )
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt_payload}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            parsed_response = json.loads(response.choices[0].message.content.strip())
            return [parsed_response.get(f"block_{i}", blocks_list[i]) for i in range(len(blocks_list))]
        except Exception as e:
            st.error(f"Groq Cloud Engine Error: {e}")
            return blocks_list

    # --- STEP 2 SITS DIRECTLY UNDER THE SELECTION ENGINE ---
    st.markdown("### 📁 Step 2: Upload Trade Documentation")
    uploaded_file = st.file_uploader("Drop invoice, packing list, or manifest here", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Run Universal Precision Translation ➔", use_container_width=True):
            with st.spinner(f"Routing documentation smoothly from {source_lang} to {target_lang}..."):
                try:
                    input_bytes = uploaded_file.read()
                    doc = pymupdf.open(stream=input_bytes, filetype="pdf")
                    
                    for page in doc:
                        text_instances = page.get_text("blocks")
                        blocks_to_translate = []
                        valid_instances = []
                        
                        for instance in text_instances:
                            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                            if text.strip() and not text.replace(".", "", 1).isdigit():
                                blocks_to_translate.append(text)
                                valid_instances.append(instance)
                        
                        if blocks_to_translate:
                            translated_blocks = translate_page_blocks
