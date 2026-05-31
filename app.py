import os
import json
import streamlit as st
import pymupdf
from openai import OpenAI

st.set_page_config(page_title="MAITAF Customs AI Lab", layout="centered")
st.title("📄 Customs PDF Translator Lab")
st.subheader("Targeted Front-End Alignment Control Node")

# --- FRONTEND VISUAL SELECTION CHANNELS ---
st.markdown("### 🌐 Step 1: Configure Language Logistics Corridor")
col1, col2 = st.columns(2)

with col1:
    source_lang = st.selectbox(
        "Translate From (Source Language):",
        ["Auto-Detect", "French", "English", "Spanish", "German", "Mandarin Chinese", "Arabic", "Portuguese", "Italian", "Dutch"]
    )

with col2:
    target_lang = st.selectbox(
        "Translate To (Target Destination Language):",
        ["English", "French", "Spanish", "German", "Mandarin Chinese", "Arabic", "Portuguese", "Italian", "Dutch"]
    )

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
            f"1. Translate all industry vocabulary accurately into {tgt}.\n"
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

    st.markdown("### 📁 Step 2: Upload Trade Documentation")
    uploaded_file = st.file_uploader("Drop invoice, packing list, or manifest here", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Run Targeted Precision Translation ➔", use_container_width=True):
            with st.spinner("Processing text coordinates with midpoint balancing..."):
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
                            translated_blocks = translate_page_blocks(blocks_to_translate, source_lang, target_lang)
                            
                            for idx, instance in enumerate(valid_instances):
                                x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                                t_text = translated_blocks[idx] if translated_blocks and idx < len(translated_blocks) else text
                                
                                page.add_redact_annot(pymupdf.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
                                page.apply_redactions()
                                
                                # Perfect baseline mid-point calculation matching main.py core algorithm
                                font_size = 8
                                center_y = y0 + ((y1 - y0) / 2) + (font_size / 3)
                                page.insert_text(pymupdf.Point(x0, center_y), t_text, fontsize=font_size, color=(0, 0, 0))
                    
                    output_bytes = doc.tobytes()
                    doc.close()
                    st.success(f"✓ Document rendered successfully from {source_lang} to {target_lang}!")
                    st.download_button(
                        label="Download Aligned PDF 📥",
                        data=output_bytes,
                        file_name=f"aligned_{target_lang}_{uploaded_file.name}",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as ex:
                    st.error(f"Fatal Engine Runtime Failure: {ex}")
