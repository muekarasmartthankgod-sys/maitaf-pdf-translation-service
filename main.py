import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pymupdf
from openai import OpenAI

app = FastAPI()

# Enable cross-origin resource sharing so your website can talk to Render securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SECURTIY HANDSHAKE BLOCK ---
# Looks for the correct key name from Render environment configurations
GROQ_TOKEN = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")

if GROQ_TOKEN:
    # Strip away accidental whitespace characters or quotation marks from copy-paste bugs
    GROQ_TOKEN = GROQ_TOKEN.strip().strip('"').strip("'")

client = OpenAI(
    api_key=GROQ_TOKEN,
    base_url="https://api.groq.com/openai/v1"
)

def translate_page_blocks(blocks_list: list) -> list:
    """Sends all text blocks on a page to Groq in a single fast, low-token batch trip."""
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
        # Utilizing the highly stable 8B model to eliminate free-tier 429 rate limits
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=0.1
        )
        raw_result = response.choices[0].message.content.strip()
        
        translated_items = [item.replace(f"ID {i}:", "").strip() for i, item in enumerate(raw_result.split("---"))]
        return translated_items
    except Exception as e:
        print(f"Llama Groq Operational Error: {e}")
        return blocks_list

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
