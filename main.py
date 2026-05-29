import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pymupdf
from google import genai
from google.genai import types

app = FastAPI()

# Standard native Google initialization pattern
# It will look for your GEMINI_API_KEY environment variable automatically
client = genai.Client()

def translate_page_blocks(blocks_list: list) -> list:
    """Sends all text blocks on a page to Gemini in one single fast trip using modern native Google SDK."""
    if not blocks_list:
        return []
    
    # Pack items efficiently into a singular prompt sequence
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
        # Using the standard SDK structural target pattern
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_payload,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )
        
        raw_result = response.text.strip()
        
        # Split items back cleanly across structural blocks
        translated_items = [item.replace(f"ID {i}:", "").strip() for i, item in enumerate(raw_result.split("---"))]
        return translated_items
        
    except Exception as e:
        print(f"Native Gemini Execution Error: {e}")
        return blocks_list  # Safely fall back to original text if something clips

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"translated_{file.filename}"
    
    # Stream data payload straight to disk
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    doc = pymupdf.open(input_path)
    
    for page in doc:
        text_instances = page.get_text("blocks")
        
        blocks_to_translate = []
        valid_instances = []
        
        # Step 1: Accumulate text instances, disregarding plain isolated metrics/digits
        for instance in text_instances:
            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
            if text.strip() and not text.replace(".", "", 1).isdigit():
                blocks_to_translate.append(text)
                valid_instances.append(instance)
        
        # Step 2: Request execution via Google AI Studio network node
        translated_blocks = translate_page_blocks(blocks_to_translate)
        
        # Step 3: Draw out redacting layers and inject corresponding English phrases
        for idx, instance in enumerate(valid_instances):
            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
            
            t_text = translated_blocks[idx] if idx < len(translated_blocks) else text
            
            page.add_redact_annot(pymupdf.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
            page.apply_redactions()
            page.insert_text(pymupdf.Point(x0, y0 + 10), t_text, fontsize=8, color=(0, 0, 0))
                
    doc.save(output_path)
    doc.close()
    
    if os.path.exists(input_path):
        os.remove(input_path)
    
    return FileResponse(output_path, media_type="application/pdf", filename=output_path)
