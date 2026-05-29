import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pymupdf  # Updated from legacy 'fitz' to match Python 3.14 specifications
from openai import OpenAI

app = FastAPI()

# Secure initialization using environment tokens
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

def translate_text(text: str, target_lang: str = "English") -> str:
    if not text.strip():
        return text
    
    try:
        # Utilizing modern OpenAI syntax execution mapping
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an expert international trade and customs compliance translator. "
                        "Translate the following text into clear, professional standard technical trade English. "
                        "Maintain legal accuracy for shipping terms, Incoterms, product descriptions, tariff headings, "
                        "and customs acronyms. Return ONLY the translated text without commentary."
                    )
                },
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Execution Error: {e}")
        return text

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"translated_{file.filename}"
    
    # Save the incoming stream
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # Open the document with modern pymupdf syntax
    doc = pymupdf.open(input_path)
    
    for page in doc:
        text_instances = page.get_text("blocks")
        
        for instance in text_instances:
            # Unpack spatial coordinate markers safely
            x0, y0, x1, y1, text, block_no, block_type = instance
            
            if text.strip():
                translated_text = translate_text(text, target_lang="English")
                
                # Apply redactions cleanly onto coordinates
                page.add_redact_annot(pymupdf.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
                page.apply_redactions()
                
                # Render clean text overlays
                page.insert_text(pymupdf.Point(x0, y0 + 10), translated_text, fontsize=9, color=(0, 0, 0))
                
    doc.save(output_path)
    doc.close()
    
    if os.path.exists(input_path):
        os.remove(input_path)
    
    return FileResponse(output_path, media_type="application/pdf", filename=output_path)
