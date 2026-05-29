import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pymupdf
from openai import OpenAI

app = FastAPI()

# Point the client directly to Google's official Gemini endpoint wrapper
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def translate_text(text: str, target_lang: str = "English") -> str:
    if not text.strip():
        return text
    
    try:
        # Utilizing the lightning-fast, highly efficient Gemini 3.5 Flash model
        response = client.chat.completions.create(
            model="gemini-3.5-flash",
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
        print(f"Gemini API Execution Error: {e}")
        return text

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"translated_{file.filename}"
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    doc = pymupdf.open(input_path)
    
    for page in doc:
        text_instances = page.get_text("blocks")
        
        for instance in text_instances:
            # Unpack spatial coordinate markers safely
            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
            
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
