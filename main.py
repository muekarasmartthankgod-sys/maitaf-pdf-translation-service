import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
from openai import OpenAI

app = FastAPI()

# Initialize the OpenAI client (it will automatically look for the OPENAI_API_KEY env variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_text(text: str, target_lang: str = "English") -> str:
    if not text.strip():
        return text
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Lightning fast, highly accurate, and incredibly cost-effective
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
            temperature=0.1 # Low temperature ensures strict, non-creative translations
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return text # Fallback to original text if API fails

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"translated_{file.filename}"
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    doc = fitz.open(input_path)
    
    for page in doc:
        text_instances = page.get_text("blocks") # Gets text and coordinates
        
        for instance in text_instances:
            x0, y0, x1, y1, text, block_no, block_type = instance
            
            if text.strip():
                # 1. Translate via OpenAI
                translated_text = translate_text(text, target_lang="English")
                
                # 2. Hide original text to prevent overlapping text layers
                page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
                page.apply_redactions()
                
                # 3. Write translated text at the exact same location
                page.insert_text(fitz.Point(x0, y0 + 10), translated_text, fontsize=9, color=(0, 0, 0))
                
    doc.save(output_path)
    doc.close()
    
    # Clean up input file
    os.remove(input_path)
    
    return FileResponse(output_path, media_type="application/pdf", filename=output_path)
