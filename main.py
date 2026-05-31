import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pymupdf
from openai import OpenAI

app = FastAPI()

# Enable cross-origin resource sharing so your website frontend can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SECURE CREDENTIAL ROUTING LAYER ---
GROQ_TOKEN = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")

if GROQ_TOKEN:
    # Automatically strip out white spaces or accidental quotes from copy-paste glitches
    GROQ_TOKEN = GROQ_TOKEN.strip().strip('"').strip("'")

client = OpenAI(
    api_key=GROQ_TOKEN,
    base_url="https://api.groq.com/openai/v1"
)

def translate_page_blocks(blocks_list: list) -> list:
    """Processes textual structures bi-directionally with high-speed LPU pipelines."""
    if not blocks_list:
        return []
    
    # Bi-directional trade-optimized system instructions
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
        # Utilizing Llama-3.1-8b-instant to maximize multi-page token limits safely
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=0.1
        )
        raw_result = response.choices[0].message.content.strip()
        
        translated_items = [item.replace(f"ID {i}:", "").strip() for i, item in enumerate(raw_result.split("---"))]
        return translated_items
    except Exception as e:
        print(f"Llama Groq Operational Error: {e}")
        return []

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"translated_{file.filename}"
    
    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        doc = pymupdf.open(input_path)
        
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
                translated_blocks = translate_page_blocks(blocks_to_translate)
                
                for idx, instance in enumerate(valid_instances):
                    x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                    
                    if translated_blocks and idx < len(translated_blocks) and translated_blocks[idx].strip():
                        t_text = translated_blocks[idx]
                    else:
                        t_text = text
                    
                    # Redact old text blocks using clean white visual canvases
                    page.add_redact_annot(pymupdf.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
                    page.apply_redactions()
                    page.insert_text(pymupdf.Point(x0, y0 + 10), t_text, fontsize=8, color=(0, 0, 0))
                    
        doc.save(output_path)
        doc.close()
        
        return FileResponse(output_path, media_type="application/pdf", filename=output_path)
        
    except Exception as error:
        print(f"Server Runtime Error: {error}")
        return {"error": str(error)}
        
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
