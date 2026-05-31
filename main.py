import os
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pymupdf
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_TOKEN = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
if GROQ_TOKEN:
    GROQ_TOKEN = GROQ_TOKEN.strip().strip('"').strip("'")

client = OpenAI(api_key=GROQ_TOKEN, base_url="https://api.groq.com/openai/v1")

def translate_page_blocks(blocks_list: list, src: str, tgt: str) -> list:
    if not blocks_list:
        return []
    
    input_manifest = {f"block_{i}": text.strip() for i, text in enumerate(blocks_list)}
    
    prompt_payload = (
        f"You are an expert multilingual international trade, logistics, and customs compliance translator.\n"
        f"Your strict operational task is to translate the values in this JSON object FROM {src} TO {tgt}.\n\n"
        "CRITICAL TRANSLATION MANDATES:\n"
        f"1. Translate all descriptive and legal industry vocabulary precisely into {tgt}.\n"
        "2. Keep all numbers, metrics, quantities, prices, and symbols EXACTLY as they are.\n"
        "3. Retain standard global logistics abbreviations (such as HS Codes, Incoterms like FOB, CIF, CIP) without alteration.\n"
        "4. Do not drop keys. Maintain the exact JSON layout configuration tracking tree structural paths.\n\n"
        f"Input Target Objective Map: {json.dumps(input_manifest, ensure_ascii=False)}"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        parsed_response = json.loads(response.choices[0].message.content.strip())
        
        reconstructed_translations = []
        for i in range(len(blocks_list)):
            translated_value = parsed_response.get(f"block_{i}", blocks_list[i])
            reconstructed_translations.append(translated_value)
            
        return reconstructed_translations
    except Exception as e:
        print(f"Llama Groq Handling Fault: {e}")
        return blocks_list

@app.post("/translate-pdf/")
async def translate_pdf(
    file: UploadFile = File(...),
    source_lang: str = Form("Auto-Detect"),  # Accepts form values sent from your web client interface
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
                if text.strip() and not text.replace(".", "", 1).isdigit():
                    blocks_to_translate.append(text)
                    valid_instances.append(instance)
            
            if blocks_to_translate:
                translated_blocks = translate_page_blocks(blocks_to_translate, source_lang, target_lang)
                
                for idx, instance in enumerate(valid_instances):
                    x0, y0, x1, y1, text, block_no, block_type = instance[:7]
                    t_text = translated_blocks[idx] if translated_blocks and idx < len(translated_blocks) else text
                    
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
