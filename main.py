import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pymupdf
from openai import OpenAI

app = FastAPI()

# Initializing the client wrapper targeting the Groq processing engine
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def translate_page_blocks(blocks_list: list) -> list:
    """Sends all text blocks on a page to Groq in a single fast batch trip."""
    if not blocks_list:
        return []
    
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
        # Route directly to the powerful Llama 3.3 70B production model
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=0.1
        )
        raw_result = response.choices[0].message.content.strip()
        
        translated_items = [item.replace(f"ID {i}:", "").strip() for i, item in enumerate(raw_result.split("---"))]
        return translated_items
    except Exception as e:
        print(f"Llama Groq Execution Error: {e}")
        return blocks_list

@app.post("/translate-pdf/")
async def translate_pdf(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"translated_{file.filename}"
    
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
        
        translated_blocks = translate_page_blocks(blocks_to_translate)
        
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
