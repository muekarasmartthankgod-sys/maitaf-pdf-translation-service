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

def translate_page_blocks(blocks_list: list) -> list:
    """Sends all text blocks on a page to Gemini in one single lightning-fast batch trip."""
    if not blocks_list:
        return []
    
    # Pack blocks into a clean structured payload with unique numeric anchors
    prompt_payload = "You are a customs translator. Translate these blocks into professional trade English. Keep industry codes/Incoterms. Return translations matching the item numbers exactly, separated by '---'. Do not add introduction text.\n\n"
    for i, text in enumerate(blocks_list):
        prompt_payload += f"ID {i}: {text.strip()}\n"
        
    try:
        response = client.chat.completions.create(
            model="gemini-3.5-flash",
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=0.1
        )
        raw_result = response.choices[0].message.content.strip()
        
        # Clean up and split the translations back into a neat list
        translated_items = [item.replace(f"ID {i}:", "").strip() for i, item in enumerate(raw_result.split("---"))]
        return translated_items
    except Exception as e:
        print(f"Gemini Speed Batch Error: {e}")
        return blocks_list  # Return original if the batch call errors out

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"translated_{file.filename}"
    
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    doc = pymupdf.open(input_path)
    
    for page in doc:
        text_instances = page.get_text("blocks")
        
        # Phase 1: Collect valid text values out of coordinates
        blocks_to_translate = []
        valid_instances = []
        
        for instance in text_instances:
            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
            if text.strip() and not text.replace(".", "", 1).isdigit(): # skip bare numbers
                blocks_to_translate.append(text)
                valid_instances.append(instance)
        
        # Phase 2: Translate the whole page text in ONE network query
        translated_blocks = translate_page_blocks(blocks_to_translate)
        
        # Phase 3: Redact and overwrite layouts sequentially
        for idx, instance in enumerate(valid_instances):
            x0, y0, x1, y1, text, block_no, block_type = instance[:7]
            
            # Ensure safe indexing fallback
            t_text = translated_blocks[idx] if idx < len(translated_blocks) else text
            
            # Overwrite layout text masks cleanly
            page.add_redact_annot(pymupdf.Rect(x0, y0, x1, y1), fill=(1, 1, 1)) 
            page.apply_redactions()
            page.insert_text(pymupdf.Point(x0, y0 + 10), t_text, fontsize=8, color=(0, 0, 0))
                
    doc.save(output_path)
    doc.close()
    
    if os.path.exists(input_path):
        os.remove(input_path)
    
    return FileResponse(output_path, media_type="application/pdf", filename=output_path)
