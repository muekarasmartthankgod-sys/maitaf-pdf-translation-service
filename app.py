import os
import json
import streamlit as st
import pymupdf
from openai import OpenAI

st.set_page_config(page_title="MAITAF Customs AI Lab", layout="centered")
st.title("📄 Customs PDF Translator Lab")
st.subheader("Targeted Routing Architecture (JSON Stable)")

# --- STREAMLIT UI LANGUAGE SELECTORS ---
st.markdown("### 🌐 Route Your Trade Corridor")
col1, col2 = st.columns(2)

with col1:
    source_lang = st.selectbox(
        "Translate From (Source):",
        ["Auto-Detect", "French", "English", "Spanish", "Mandarin Chinese", "German", "Arabic", "Portuguese", "Japanese", "Hindi", "Italian", "Dutch", "Vietnamese", "Polish"]
    )

with col2:
    target_lang = st.selectbox(
        "Translate To (Target):",
        ["English", "French", "Spanish", "Mandarin Chinese", "German", "Arabic", "Portuguese", "Japanese", "Hindi", "Italian", "Dutch", "Vietnamese", "Polish"]
    )

# --- TESTING UTILITY: AUTO-GENENERATOR ---
def make_sample_pdf():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((40, 50), "FACTURE COMMERCIALE", fontsize=22, fontname="helv-bold", color=(0.06, 0.09, 0.16))
    page.insert_text((40, 95), "ÉMETTEUR (Vendeur) :", fontsize=10, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((40, 110), "Logistique Transatlantique SAS\n12 Rue de l'Arsenal, 75004 Paris, France", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    page.insert_text((320, 95), "DESTINATAIRE (Acheteur) :", fontsize=10, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((320, 110), "MAITAF Trade Compliance Ltd\n85 Great Portland Street, London, UK", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    page.insert_text((40, 185), "Numéro de Facture :  FC-2026-0599", fontsize=9, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.draw_rect(pymupdf.Rect(40, 215, 572, 232), color=(0.95, 0.96, 0.98), fill=(0.95, 0.96, 0.98))
    page.insert_text((45, 227), "Poste   Code SH      Désignation des Marchandises                                Qté      Total HT", fontsize=9, fontname="helv-bold", color=(0.06, 0.09, 0.16))
    page.insert_text((45, 250), "01         8471.30      Ordinateurs portables professionnels (MAITAF Node X)   15       12 750,00 €", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    page.insert_text((380, 290), "Net à Payer :       12 750,00 €", fontsize=10, fontname="helv-bold", color=(0.06, 0.09, 0.16))
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

st.sidebar.header("Testing Tools Workspace")
st.sidebar.download_button(
    label="Download Sample French Invoice 📄",
    data=make_sample_pdf(),
    file_name="facture_francaise_test.pdf",
    mime="application/pdf",
    use_container_width=True
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
        
        # Enforcing precise target parameter routing instructions
        prompt_payload = (
            f"You are an expert multilingual international trade, logistics, and customs compliance translator.\n"
            f"Your strict operational task is to translate the values in this JSON object FROM {src} TO {tgt}.\n\n"
            "CRITICAL TRANSLATION MANDATES:\n"
            f"1. Translate all descriptive and legal industry vocabulary precisely into {tgt}.\n"
            "2. Keep all numbers, metrics, quantities, prices, and symbols EXACTLY as they are.\n"
            "3. Retain standard global logistics abbreviations (such as HS Codes, Incoterms like FOB, CIF, CIP) without alteration.\n"
            "4. Do not drop keys. Maintain the exact JSON layout configuration tracking tree structural paths.\n\n"
            f"Input Target Objective Map: {json.dumps(input_manifest, ensure_ascii=False)}"
        )
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt_payload}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            parsed_response = json.loads(response.choices[0].message.content.strip())
            
            reconstructed_translations = []
            for i in range(len(blocks_list)):
                translated_value = parsed_response.get(f"block_{i}", blocks_list[i])
                reconstructed_translations.append(translated_value)
                
            return reconstructed_translations
        except Exception as e:
            st.error(f"Groq Cloud Engine Error: {e}")
            return blocks_list

    uploaded_file = st.file_uploader("Upload a Customs Document, Invoice or Manifest (PDF)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Run Targeted Translation Sequence ➔", use_container_width=True):
            with st.spinner("Executing custom linguistic corridor routing..."):
                try:
                    input_bytes = uploaded_file.read()
                    doc = pymupdf.open(stream=input_bytes, filetype="pdf")
                    status_text = st.empty()
                    
                    for page_num, page in enumerate(doc):
                        status_text.text(f"Processing structural layouts on Page {page_num + 1}...")
                        text_instances = page.get_text("blocks")
                        blocks_to_translate = []
                        valid_instances = []
                        
                        for instance in text_instances:
                            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                            if text.strip() and not text.replace(".", "", 1).isdigit():
                                blocks_to_translate.append(text)
                                valid_instances.append(instance)
                        
                        if blocks_to_translate:
                            # Pass user language selections down into processing loop arrays
                            translated_blocks = translate_page_blocks(blocks_to_translate, source_lang, target_lang)
                            
                            for idx, instance in enumerate(valid_instances):
                                x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                                t_text = translated_blocks[idx] if translated_blocks and idx < len(translated_blocks) else text
                                
                                page.add_redact_annot(pymupdf.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
                                page.apply_redactions()
                                page.insert_text(pymupdf.Point(x0, y0 + 10), t_text, fontsize=8, color=(0, 0, 0))
                    
                    status_text.empty()
                    output_bytes = doc.tobytes()
                    doc.close()
                    
                    st.success(f"✓ Document routed cleanly from {source_lang} to {target_lang}!")
                    st.download_button(
                        label="Download Translated PDF 📥",
                        data=output_bytes,
                        file_name=f"directed_{target_lang}_{uploaded_file.name}",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as ex:
                    st.error(f"Fatal Engine Runtime Failure: {ex}")
