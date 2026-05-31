import os
import json
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
    GROQ_TOKEN = GROQ_TOKEN.strip().strip('"').strip("'")

client = OpenAI(
    api_key=GROQ_TOKEN,
    base_url="https://api.groq.com/openai/v1"
)

def translate_page_blocks(blocks_list: list) -> list:
    """Sends page elements to Groq using strict JSON schema tracking to eliminate alignment drift."""
    if not blocks_list:
        return []
    
    # Construct a clean JSON-stringified map template for the payload
    input_manifest = {f"block_{i}": text.strip() for i, text in enumerate(blocks_list)}
    
    prompt_payload = (
        "You are an expert multilingual international trade, logistics, and customs compliance translation engine.\n"
        "Your sole task is to translate the text values in this JSON object while maintaining strict dictionary structures.\n\n"
        "TRANSLATION MATRIX RULES:\n"
        "1. If a value is in a foreign language (French, Spanish, Mandarin, etc.), translate it to technical trade English.\n"
        "2. If a value is ALREADY in English, translate it into professional standard French (or the designated customs zone language).\n"
        "3. Maintain all numeric references, HS codes, and Incoterms (FOB, CIP, CIF) exactly.\n"
        "4. DO NOT alter raw numeric values or standalone digits.\n\n"
        "CRITICAL: Return ONLY a valid JSON object matching the exact keys provided. No introductory text, no explanations.\n"
        f"Input Target Objective: {json.dumps(input_manifest, ensure_ascii=False)}"
    )
    
    try:
        # Utilizing Llama 3.1 8B with temperature 0.0 for strict, deterministic output matching
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=0.0,
            response_format={"type": "json_object"}  # Forces Groq to return a perfect JSON tree
        )
        
        parsed_response = json.loads(response.choices[0].message.content.strip())
        
        # Build a safe array output matching the original document sequence order
        reconstructed_translations = []
        for i in range(len(blocks_list)):
            translated_value = parsed_response.get(f"block_{i}", blocks_list[i])
            reconstructed_translations.append(translated_value)
            
        return reconstructed_translations
    except Exception as e:
        print(f"Secure Translation Handshake Bypass Triggered: {e}")
        return blocks_list  # Safe fallback: keeps original text if the API encounters an error

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
