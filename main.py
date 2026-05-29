import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
import requests

app = FastAPI()

# Example translation function using DeepL API
def translate_text(text: str, target_lang: str = "EN") -> str:
    if not text.strip():
        return text
    
    DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
    url = "https://api-free.deepl.com/v2/translate"
    
    payload = {
        "text": [text],
        "target_lang": target_lang
    }
    headers = {
        "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        return response.json()["translations"][0]["text"]
    except Exception:
        return text # Fallback to original text if translation fails

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    input_path = f"temp_{file.filename}"
    output_path = f"translated_{file.filename}"
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # Process PDF
    doc = fitz.open(input_path)
    
    for page in doc:
        text_instances = page.get_text("blocks") # Gets text and coordinates
        
        for instance in text_instances:
            x0, y0, x1, y1, text, block_no, block_type = instance
            
            if text.strip():
                # 1. Translate the block text
                translated_text = translate_text(text, target_lang="EN")
                
                # 2. Redact/Hide original text to prevent overlapping
                page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
                page.apply_redactions()
                
                # 3. Insert translated text at the exact same location
                page.insert_text(fitz.Point(x0, y0 + 10), translated_text, fontsize=9, color=(0, 0, 0))
                
    doc.save(output_path)
    doc.close()
    
    # Clean up input file
    os.remove(input_path)
    
    # Return translated PDF to user
    return FileResponse(output_path, media_type="application/pdf", filename=output_path)
