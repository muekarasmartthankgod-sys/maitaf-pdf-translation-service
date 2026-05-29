import os
import streamlit as st
import pymupdf
from openai import OpenAI

# Page configurations for a professional, clean user layout
st.set_page_config(page_title="MAITAF Customs AI Lab", layout="centered")
st.title("📄 Customs PDF Translator Lab")
st.subheader("Production Node via Groq (Llama 3.3)")

# --- BULLETPROOF API KEY LOOKUP STRATEGY ---
api_key = None

if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
elif os.environ.get("GROQ_API_KEY"):
    api_key = os.environ.get("GROQ_API_KEY")

# Automatically strip out white spaces, accidental quotes or brackets from clipboard issues
if api_key:
    api_key = api_key.strip().strip('"').strip("'")

if not api_key:
    st.error("❌ GROQ_API_KEY is missing!")
    st.markdown(
        """
        **How to fix this:**
        1. Go to your Streamlit App Dashboard (**share.streamlit.io**).
        2. Click the three dots `...` next to your app and choose **Settings** -> **Secrets**.
        3. Paste your key exactly like this:
        ```toml
        GROQ_API_KEY = "gsk_your_actual_key_here"
        ```
        4. Click **Save**.
        """
    )
else:
    # Initialize OpenAI client structural wrapper mapping to Groq endpoints
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    def translate_page_blocks(blocks_list: list) -> list:
        if not blocks_list:
            return []
        
        prompt_payload = (
            "You are an expert international trade and customs compliance translator.\n"
            "Translate these isolated text blocks into professional standard technical trade English.\n"
            "Maintain legal accuracy for shipping terms, Incoterms, product descriptions, tariff headings, "
            "and customs acronyms. Return translations matching the item numbers exactly, separated by '---'.\n"
            "Do not include any introductions, conclusions, or extra explanations.\n\n"
        )
        
        for i, text in enumerate(blocks_list):
            prompt_payload += f"ID {i}: {text.strip()}\n"
            
        try:
            # Utilizing Llama 3.3 70B Versatile for institutional customs mapping accuracy
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt_payload}],
                temperature=0.1
            )
            raw_result = response.choices[0].message.content.strip()
            
            translated_items = [item.replace(f"ID {i}:", "").strip() for i, item in enumerate(raw_result.split("---"))]
            return translated_items
        except Exception as e:
            st.error(f"Groq Cloud Engine Error: {e}")
            return blocks_list

    # Native Drag & Drop UI Component Block
    uploaded_file = st.file_uploader("Upload a foreign Customs Document, Invoice or Manifest (PDF)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Run Translation Sequence ➔", use_container_width=True):
            with st.spinner("Analyzing document structure mapping... Please wait."):
                try:
                    # Stream directly out of uploaded binary buffer state mapping
                    input_bytes = uploaded_file.read()
                    doc = pymupdf.open(stream=input_bytes, filetype="pdf")
                    
                    status_text = st.empty()
                    
                    for page_num, page in enumerate(doc):
                        status_text.text(f"Scanning text element arrays on Page {page_num + 1}...")
                        text_instances = page.get_text("blocks")
                        blocks_to_translate = []
                        valid_instances = []
                        
                        for instance in text_instances:
                            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                            # Cleanly skip plain strings that are isolated numbers/metrics
                            if text.strip() and not text.replace(".", "", 1).isdigit():
                                blocks_to_translate.append(text)
                                valid_instances.append(instance)
                        
                        if blocks_to_translate:
                            status_text.text(f"Translating {len(blocks_to_translate)} elements on Page {page_num + 1} via Groq...")
                            translated_blocks = translate_page_blocks(blocks_to_translate)
                            
                            # Layout-preserved overlay sequence
                            for idx, instance in enumerate(valid_instances):
                                x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                                
                                # Evaluates safe text extraction with proper structural fallback matching
                                t_text = translated_blocks[idx] if idx < len(translated_blocks) else text
                                
                                # Overwrite old text cleanly using white block masks
                                page.add_redact_annot(pymupdf.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
                                page.apply_redactions()
                                page.insert_text(pymupdf.Point(x0, y0 + 10), t_text, fontsize=8, color=(0, 0, 0))
                    
                    status_text.empty()
                    output_bytes = doc.tobytes()
                    doc.close()
                    
                    st.success("✓ Translation Pipeline execution complete!")
                    
                    st.download_button(
                        label="Download Translated English PDF 📥",
                        data=output_bytes,
                        file_name=f"translated_{uploaded_file.name}",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as ex:
                    st.error(f"Fatal Engine Runtime Failure: {ex}")
