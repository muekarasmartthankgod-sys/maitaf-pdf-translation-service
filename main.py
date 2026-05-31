import os
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pymupdf
from openai import OpenAI

app = FastAPI()

# Enable cross-origin resource sharing for your website domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SECURE API KEY ROUTING ---
GROQ_TOKEN = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
if GROQ_TOKEN:
    GROQ_TOKEN = GROQ_TOKEN.strip().strip('"').strip("'")

client = OpenAI(api_key=GROQ_TOKEN, base_url="https://api.groq.com/openai/v1")

def translate_page_blocks(blocks_list: list, src: str, tgt: str) -> list:
    if not blocks_list:
        return []
    
    # Map the text fragments to a strict JSON structure
    input_manifest = {f"block_{i}": text.strip() for i, text in enumerate(blocks_list)}
    
    prompt_payload = (
        f"You are an expert multilingual international trade, logistics, and customs compliance translator.\n"
        f"Your strict operational task is to translate the values in this JSON object FROM {src} TO {tgt}.\n\n"
        "CRITICAL TRANSLATION MANDATES:\n"
        f"1. Translate all descriptive, logistical, and legal industry vocabulary precisely into {tgt}.\n"
        "2. Keep all numbers, metrics, quantities, prices, dates, and currency symbols EXACTLY as they are.\n"
        "3. Retain standard global logistics abbreviations (such as HS Codes, Incoterms like FOB, CIF, CIP) without alteration.\n"
        "4. Do not drop keys. Return ONLY a valid JSON object matching the exact input structure layout.\n\n"
        f"Input Target Map: {json.dumps(input_manifest, ensure_ascii=False)}"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=0.0,
            response_format={"type": "json_object"}  # Hard-locks Groq into a clean JSON tree return
        )
        parsed_response = json.loads(response.choices[0].message.content.strip())
        return [parsed_response.get(f"block_{i}", blocks_list[i]) for i in range(len(blocks_list))]
    except Exception as e:
        print(f"Universal Router API Error: {e}")
        return blocks_list

@app.post("/translate-pdf/")
async def translate_pdf(
    file: UploadFile = File(...),
    source_lang: str = Form("Auto-Detect"),  
    target_lang: str = Form("English")      
):
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
                # Filter out pure numbers to preserve columns from scrambling
                if text.strip() and not text.replace(".", "", 1).isdigit():
                    blocks_to_translate.append(text)
                    valid_instances.append(instance)
            
            if blocks_to_translate:
                translated_blocks = translate_page_blocks(blocks_to_translate, source_lang, target_lang)
                
                for idx, instance in enumerate(valid_instances):
                    x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                    t_text = translated_blocks[idx] if translated_blocks and idx < len(translated_blocks) else text
                    
                    # 1. Redact the old text using a clean white canvas mask
                    rect = pymupdf.Rect(x0, y0, x1, y1)
                    page.add_redact_annot(rect, fill=(1, 1, 1)) 
                    page.apply_redactions()
                    
                    # 2. DYNAMIC COORDINATE SAFETY ZONE
                    # If the block starts on the left side of the page (description column),
                    # cap its width at x=380 so it can never bleed into numbers on the right.
                    if x0 < 300 and x1 > 400:
                        render_rect = pymupdf.Rect(x0, y0, 380, y1 + 15)
                    else:
                        render_rect = pymupdf.Rect(x0, y0, x1, y1 + 10)
                    
                    # 3. Draw text with forced text-wrapping boundaries
                    page.insert_textbox(render_rect, t_text, fontsize=8, fontname="helv", color=(0, 0, 0))
                    
        doc.save(output_path)
        doc.close()
        return FileResponse(output_path, media_type="application/pdf", filename=output_path)
        
    except Exception as error:
        print(f"Universal Server Runtime Error: {error}")
        return {"error": str(error)}
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
