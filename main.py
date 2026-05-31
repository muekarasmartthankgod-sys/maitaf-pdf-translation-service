import os
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pymupdf
import pdfplumber
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

def translate_payload_dictionary(payload: dict, src: str, tgt: str) -> dict:
    if not payload:
        return {}
    
    prompt_payload = (
        f"You are an expert multilingual international trade, logistics, and customs compliance translator.\n"
        f"Translate the text values inside this JSON object from {src} into {tgt}.\n\n"
        "STRICT MANDATES:\n"
        f"1. Translate all industry descriptions, legal disclaimers, and headers into {tgt}.\n"
        "2. Keep all numbers, metrics, quantities, prices, dates, currency symbols, and bank codes EXACTLY as they are.\n"
        "3. Maintain standard logistics shorthand (such as HS Codes, Incoterms like FOB, CIP, CIF).\n"
        "4. Return ONLY a valid JSON object matching the exact input keys perfectly. No introductions, no explanations.\n\n"
        f"Target Objective JSON: {json.dumps(payload, ensure_ascii=False)}"
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
        print(f"API Translation Loop Error: {e}")
        return payload

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
            
        # Initialize dual extraction engines
        pymupdf_doc = pymupdf.open(input_path)
        
        with pdfplumber.open(input_path) as plumber_doc:
            for page_idx, plumber_page in enumerate(plumber_doc):
                pymupdf_page = pymupdf_doc[page_idx]
                
                # 1. Isolate and lock down table structures to protect rows from colliding
                tables = plumber_page.find_tables()
                table_areas = []
                
                translation_manifest = {}
                element_index = 0
                
                # 2. Extract content cell-by-cell out of data matrix frameworks
                for table in tables:
                    table_areas.append(table.bbox)  # (x0, y0, x1, y1) bounds map
                    table_data = table.extract()
                    
                    for row_idx, row in enumerate(table_data):
                        for col_idx, cell_text in enumerate(row):
                            if cell_text and cell_text.strip():
                                # Check if it's a structural text wrapper and not a bare standalone number
                                if not cell_text.strip().replace(",", "", 1).replace(".", "", 1).replace("€", "").strip().isdigit():
                                    key = f"table_{element_index}"
                                    translation_manifest[key] = cell_text.strip()
                                    element_index += 1
                
                # 3. Extract all loose text blocks outside table areas
                blocks = pymupdf_page.get_text("blocks")
                loose_blocks = []
                
                for b in blocks:
                    bx0, by0, bx1, by1, b_text, b_no, b_type = b[:7]
                    if b_text.strip():
                        # Verify the block does not overlap an extracted table grid
                        in_table = False
                        for t_bbox in table_areas:
                            tx0, ty0, tx1, ty1 = t_bbox
                            if not (bx1 < tx0 or bx0 > tx1 or by1 < ty0 or by0 > ty1):
                                in_table = True
                                break
                        
                        if not in_table and not b_text.strip().replace(".", "", 1).isdigit():
                            key = f"loose_{element_index}"
                            translation_manifest[key] = b_text.strip()
                            loose_blocks.append((key, b))
                            element_index += 1
                
                # 4. Fire structured multi-language payload translation trip
                translated_data = translate_payload_dictionary(translation_manifest, source_lang, target_lang)
                
                # 5. Overwrite Non-Table Text Layers Safely
                for key, b_info in loose_blocks:
                    bx0, by0, bx1, by1, b_text, b_no, b_type = b_info[:7]
                    t_text = translated_data.get(key, b_text)
                    
                    pymupdf_page.add_redact_annot(pymupdf.Rect(bx0, by0, bx1, by1), fill=(1, 1, 1))
                    pymupdf_page.apply_redactions()
                    pymupdf_page.insert_textbox(pymupdf.Rect(bx0, by0, bx1, by1 + 10), t_text, fontsize=8, fontname="helv", color=(0, 0, 0))
                
                # 6. Overwrite Table Text Cells without shifting accounting lines
                cell_element_idx = 0
                for table in tables:
                    # Clean the entire background architecture of the old table once
                    tx0, ty0, tx1, ty1 = table.bbox
                    
                    table_data = table.extract()
                    # Iterate through extracted cells to overlay values onto spatial layout centers
                    for row_idx, row in enumerate(table_data):
                        for col_idx, cell_text in enumerate(row):
                            if cell_text and cell_text.strip():
                                key = f"table_{cell_element_idx}"
                                t_text = translated_data.get(key, cell_text.strip()) if key in translation_manifest else cell_text
                                
                                # Find precise structural coordinate bounding boxes for individual cells
                                cell_obj = table.cells[row_idx][col_idx]
                                if cell_obj:
                                    cx0, cy0, cx1, cy1 = cell_obj
                                    
                                    # Clear old text trace layers within cell grid constraints
                                    pymupdf_page.add_redact_annot(pymupdf.Rect(cx0 + 2, cy0 + 2, cx1 - 2, cy1 - 2), fill=(1, 1, 1))
                                    pymupdf_page.apply_redactions()
                                    
                                    # Write back cell translations with forced boundary word wrapping
                                    pymupdf_page.insert_textbox(pymupdf.Rect(cx0 + 3, cy0 + 3, cx1 - 3, cy1), t_text, fontsize=7.5, fontname="helv", color=(0, 0, 0))
                                
                                cell_element_idx += 1
                                
        pymupdf_doc.save(output_path)
        pymupdf_doc.close()
        return FileResponse(output_path, media_type="application/pdf", filename=output_path)
        
    except Exception as err:
        print(f"Cellular Extraction Execution Critical Failure: {err}")
        return {"error": str(err)}
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
