import os
import streamlit as st
import pymupdf
from openai import OpenAI

# Page configurations for a professional, clean user layout
st.set_page_config(page_title="MAITAF Customs AI Lab", layout="centered")
st.title("📄 Customs PDF Translator Lab")
st.subheader("Production Node via Groq (Llama 3.1 High-Permissive)")

# --- TESTING UTILITY: AUTO-GENERATE FRENCH INVOICE ON THE FLY ---
def make_sample_pdf():
    """Generates a clean French Commercial Invoice layout programmatically using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    
    # Header Elements
    page.insert_text((40, 50), "FACTURE COMMERCIALE", fontsize=22, fontname="helv-bold", color=(0.06, 0.09, 0.16))
    
    # Address Headers Matrix
    page.insert_text((40, 95), "ÉMETTEUR (Vendeur) :", fontsize=10, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((40, 110), "Logistique Transatlantique SAS\n12 Rue de l'Arsenal, 75004 Paris, France\nN° TVA : FR 89 123 456 789\nEORI : FR123456789000", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    
    page.insert_text((320, 95), "DESTINATAIRE (Acheteur) :", fontsize=10, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((320, 110), "MAITAF Trade Compliance Ltd\n85 Great Portland Street, London, W1W 7LT, UK\nGB EORI : GB987654321000", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    
    # Metadata Block
    page.insert_text((40, 185), "Numéro de Facture :  FC-2026-0599", fontsize=9, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((40, 200), "Date d'Émission :     31 Mai 2026", fontsize=9, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((40, 215), "Conditions :               CIP London Port (Incoterms 2020)", fontsize=9, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((40, 230), "Devise :                     EUR (€)", fontsize=9, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    
    # Line Items Table Setup
    page.draw_rect(pymupdf.Rect(40, 255, 572, 272), color=(0.95, 0.96, 0.98), fill=(0.95, 0.96, 0.98))
    page.insert_text((45, 267), "Poste   Code SH      Désignation des Marchandises                                Qté      Total HT", fontsize=9, fontname="helv-bold", color=(0.06, 0.09, 0.16))
    
    page.insert_text((45, 290), "01         8471.30      Ordinateurs portables professionnels (MAITAF Node X)   15       12 750,00 €", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    page.insert_text((45, 310), "02         8504.40      Convertisseurs électriques statiques (Alimentation)       30         1 350,00 €", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    page.insert_text((45, 330), "03         8544.42      Câbles de connexion avec pièces de fixation                   50            625,00 €", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    
    # Totals Layer
    page.draw_line(pymupdf.Point(40, 350), pymupdf.Point(572, 350), color=(0.8, 0.83, 0.88), width=1)
    page.insert_text((380, 365), "Montant Total HT :", fontsize=9, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((490, 365), "14 725,00 €", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    page.insert_text((380, 380), "Net à Payer :", fontsize=10, fontname="helv-bold", color=(0.06, 0.09, 0.16))
    page.insert_text((490, 380), "14 725,00 €", fontsize=10, fontname="helv-bold", color=(0.06, 0.09, 0.16))
    
    # Declarations Footer
    page.insert_text((40, 420), "Spécifications d'Expédition", fontsize=10, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((40, 435), "• Nombre total de colis : 2 Palettes Filmées\n• Poids Net Total : 180 kg   |   Poids Brut Total : 215 kg", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.33))
    page.insert_text((40, 480), "Déclaration en Douane", fontsize=10, fontname="helv-bold", color=(0.12, 0.16, 0.23))
    page.insert_text((40, 495), "L'exportateur des produits couverts par le présent document déclare que, sauf indication\nclaire du contraire, ces produits ont l'origine préférentielle Union Européenne (FR).", fontsize=9, fontname="helv-ital", color=(0.3, 0.35, 0.43))
    
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

# Render testing layout utilities inside sidebar
st.sidebar.header("Testing Tools Workspace")
st.sidebar.markdown("Need a fast multi-block document to verify your Groq translation loops?")
st.sidebar.download_button(
    label="Download Sample French Invoice 📄",
    data=make_sample_pdf(),
    file_name="facture_francaise_test.pdf",
    mime="application/pdf",
    use_container_width=True
)

# --- BULLETPROOF API KEY LOOKUP STRATEGY ---
api_key = None

if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
elif os.environ.get("GROQ_API_KEY"):
    api_key = os.environ.get("GROQ_API_KEY")

if api_key:
    api_key = api_key.strip().strip('"').strip("'")

if not api_key:
    st.error("❌ GROQ_API_KEY is missing in your App Secrets!")
else:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    def translate_page_blocks(blocks_list: list) -> list:
        if not blocks_list:
            return []
        
        prompt_payload = (
            "You are an expert multilingual international trade, logistics, and customs compliance translator.\n"
            "Your task is to translate the provided text blocks cleanly while strictly maintaining legal and technical accuracy.\n\n"
            "DYNAMIC ROUTING RULES:\n"
            "1. If the text block is in a foreign language (e.g., French, Mandarin, Spanish, German, etc.), translate it into professional standard technical trade English.\n"
            "2. If the text block is ALREADY in English, translate it into the corresponding target foreign trade language required for the customs zone.\n"
            "3. Preserve all technical acronyms, HS codes, Incoterms (CIP, FOB, EXW), and numbers exactly.\n\n"
            "Maintain legal accuracy for shipping terms, product descriptions, tariff headings, and logistics metrics.\n"
            "Return translations matching the item IDs exactly, separated by '---'.\n"
            "Do not include any introductions, conclusions, or extra explanations.\n\n"
        )
        
        for i, text in enumerate(blocks_list):
            prompt_payload += f"ID {i}: {text.strip()}\n"
            
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt_payload}],
                temperature=0.1
            )
            raw_result = response.choices[0].message.content.strip()
            
            translated_items = [item.replace(f"ID {i}:", "").strip() for i, item in enumerate(raw_result.split("---"))]
            return translated_items
        except Exception as e:
            st.error(f"Groq Cloud Engine Error: {e}")
            return []

    # UI Core File Drop Component Block
    uploaded_file = st.file_uploader("Upload a foreign Customs Document, Invoice or Manifest (PDF)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Run Translation Sequence ➔", use_container_width=True):
            with st.spinner("Analyzing document structure mapping... Please wait."):
                try:
                    input_bytes = uploaded_file.read()
                    doc = pymupdf.open(stream=input_bytes, filetype="pdf")
                    
                    status_text = st.empty()
                    
                    for page_num, page in enumerate(doc):
                        status_text.text(f"Scanning elements on Page {page_num + 1}...")
                        text_instances = page.get_text("blocks")
                        blocks_to_translate = []
                        valid_instances = []
                        
                        for instance in text_instances:
                            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                            if text.strip() and not text.replace(".", "", 1).isdigit():
                                blocks_to_translate.append(text)
                                valid_instances.append(instance)
                        
                        if blocks_to_translate:
                            status_text.text(f"Translating Page {page_num + 1} via Groq LPU engine...")
                            translated_blocks = translate_page_blocks(blocks_to_translate)
                            
                            for idx, instance in enumerate(valid_instances):
                                x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                                
                                if translated_blocks and idx < len(translated_blocks) and translated_blocks[idx].strip():
                                    t_text = translated_blocks[idx]
                                else:
                                    t_text = text
                                
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
