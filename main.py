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

def translate_structured_payload(text_lines_dict: dict, src: str, tgt: str) -> dict:
    """Translates whole document line-strips using strict JSON tree matching."""
    if not text_lines_dict:
        return {}
    
    prompt_payload = (
        f"You are an expert multilingual international trade, logistics, and customs compliance translation engine.\n"
        f"Your task is to translate the values inside this JSON object from {src} into {tgt}.\n\n"
        "STRICT COMPLIANCE DIRECTIVES:\n"
        f"1. Translate all descriptive phrases, table headers, and legal disclaimers accurately into {tgt}.\n"
        "2. Keep all numbers, internal codes, IDs, quantities, prices, dates, bank IBANs, and currency symbols EXACTLY as they are.\n"
        "3. Preserve all standard trade abbreviations (HS Codes, Incoterms, VAT, TTC, HT).\n"
        "4. Return ONLY a valid JSON object matching the input keys perfectly. Do not add introductory or explanatory text.\n\n"
        f"Target Objective JSON: {json.dumps(text_lines_dict, ensure_ascii=False)}"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_payload}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Llama translation communication failure: {e}")
        return text_lines_dict

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
            # Step 1: Extract every word with its explicit spatial coordinate tracking matrix
            words = page.get_text("words") 
            
            # Step 2: Sort words vertically by row baseline coordinate line layers (y1 coordinate index 3)
            # Grouping items sitting on the exact same horizontal plane line blocks
            lines_map = {}
            for w in words:
                x0, y0, x1, y1, text, block_no, line_no, word_no = w
                # Create a baseline grouping key using an integer tolerance threshold segment
                line_key = int(y1)
                
                # Bundle words sharing proximity on the vertical grid
                found_key = None
                for existing_key in lines_map.keys():
                    if abs(existing_key - line_key) <= 3:  # 3-point visual line tolerance tracking
                        found_key = existing_key
                        break
                        
                if found_key is not None:
                    lines_map[found_key].append(w)
                else:
                    lines_map[line_key] = [w]
            
            # Step 3: Sort words inside each horizontal row from left to right (x0 coordinate index 0)
            sorted_lines_payload = {}
            line_index = 0
            for k in sorted(lines_map.keys()):
                row_words = sorted(lines_map[k], key=lambda x: x[0])
                full_line_string = " ".join([word[4] for word in row_words])
                
                if full_line_string.strip():
                    # Save spatial coordinates data matching whole horizontal row box envelopes
                    min_x = min([word[0] for word in row_words])
                    max_x = max([word[2] for word in row_words])
                    min_y = min([word[1] for word in row_words])
                    max_y = max([word[3] for word in row_words])
                    
                    sorted_lines_payload[f"line_{line_index}"] = {
                        "text": full_line_string,
                        "coords": (min_x, min_y, max_x, max_y)
                    }
                    line_index += 1
            
            # Step 4: Extract string segments for language pipeline batch conversion tracking
            translation_input = {key: val["text"] for key, val in sorted_lines_payload.items()}
            translated_output = translate_structured_payload(translation_input, source_lang, target_lang)
            
            # Step 5: Execute structural visual vector mask cleaning and write back translated items
            for key, val in sorted_lines_payload.items():
                x0, y0, x1, y1 = val["coords"]
                t_text = translated_output.get(key, val["text"])
                
                # Apply high-precision white out blocking over the old structural lines canvas envelope
                page.add_redact_annot(pymupdf.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=(1, 1, 1))
                page.apply_redactions()
                
                # Insert the translated row text exactly where the old line lived
                page.insert_text(pymupdf.Point(x0, y1), t_text, fontsize=8, fontname="helv", color=(0, 0, 0))
                
        doc.save(output_path)
        doc.close()
        return FileResponse(output_path, media_type="application/pdf", filename=output_path)
        
    except Exception as err:
        print(f"API Grid Translation Critical Failure: {err}")
        return {"error": str(err)}
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
