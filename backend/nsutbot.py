import os
import faiss
import pickle
import time
import shutil
import uuid
import subprocess
import platform
import base64
import urllib.parse
import secrets
from io import BytesIO
from pathlib import Path
from typing import List, Generator, Dict, Optional
import uvicorn

# --- Web Framework & API Imports ---
from fastapi import FastAPI, Response, HTTPException, Request, status, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- Auth & AI Imports ---
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium
from sentence_transformers import SentenceTransformer
from groq import Groq

# Try importing docx2pdf (Optional dependency)
try:
    from docx2pdf import convert as docx_convert
    DOCX2PDF_AVAILABLE = True
except ImportError:
    DOCX2PDF_AVAILABLE = False

# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
load_dotenv()

app = FastAPI()

# Credentials & URLs
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Models
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_CHAT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384 

# Directories
DB_DIR = Path("vector_db")
DB_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# File Paths
INDEX_PATH = DB_DIR / "index.faiss"
CHUNKS_PATH = DB_DIR / "chunks.pkl"

# 2. CLIENT INITIALIZATION
# ---------------------------------------------------------

bot_keys = [
    os.getenv("BOT1"),
    os.getenv("BOT2"),
    os.getenv("BOT3"),
    os.getenv("BOT4"),
    os.getenv("BOT5")
]

valid_bot_clients = [Groq(api_key=key) for key in bot_keys if key]
if not valid_bot_clients:
    raise ValueError("No valid Groq API keys found in environment variables.")

current_bot_index = 0

def get_next_bot_client():
    global current_bot_index
    selected_bot = valid_bot_clients[current_bot_index]
    current_bot_index = (current_bot_index + 1) % len(valid_bot_clients)
    return selected_bot

embedder = SentenceTransformer(EMBEDDING_MODEL)

# 3. DATABASE LOADING
# ---------------------------------------------------------
if INDEX_PATH.exists() and CHUNKS_PATH.exists():
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {len(chunks)} chunks from disk.")
else:
    index = faiss.IndexFlatL2(DIMENSION)
    chunks = []

# In-memory session store
sessions = {}

# 4. MIDDLEWARE
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. SECURITY DEPENDENCY (THE GATEKEEPER)
# ---------------------------------------------------------
def get_current_user(request: Request):
    """
    Checks if a valid session token exists in the cookie.
    If yes: Returns the user data.
    If no: Raises a 401 error immediately (Stopping the request).
    """
    token = request.cookies.get("auth_token")
    
    if not token:
        # No cookie found
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Authentication required"
        )
    
    if token not in sessions:
        # Cookie exists but is invalid/expired (server restarted)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Session expired"
        )
        
    return sessions[token]

# 6. DATA MODELS
# ---------------------------------------------------------
class LoginRequest(BaseModel):
    token: str

class MessageRequest(BaseModel):
    message: str
    context: List[Dict[str, str]] = []
    file: Optional[str] = None

# 7. HELPER FUNCTIONS
# ---------------------------------------------------------
def chunk_text(text: str, size: int = 500, overlap: int = 50) -> Generator[str, None, None]:
    words = text.split()
    if len(words) <= size:
        yield text
        return
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i + size])

def save_db():
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def encode_image(pil_image):
    buffered = BytesIO()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def any_to_images(input_path: Path, output_dir: Path) -> List[Image.Image]:
    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    
    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        return [Image.open(input_path)]

    if ext in [".txt", ".md", ".csv", ".py", ".js", ".json"]:
        try:
            text = input_path.read_text(encoding="utf-8", errors="ignore")
            img = Image.new("RGB", (1200, 1600), "white")
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
            draw.text((20, 20), text[:4000], fill="black", font=font)
            return [img]
        except Exception:
            return []

    if ext == ".docx":
        pdf_path = input_path.with_suffix(".pdf")
        if DOCX2PDF_AVAILABLE and (platform.system() == "Windows" or platform.system() == "Darwin"):
            try:
                docx_convert(str(input_path), str(pdf_path))
            except Exception:
                pass
        if not pdf_path.exists():
            cmd = ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(input_path)]
            if os.name == 'nt': cmd[0] = "soffice"
            subprocess.run(cmd, check=True)
        input_path = pdf_path 

    if ext in [".pptx", ".ppt", ".xlsx", ".xls"]:
        cmd = ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(input_path)]
        if os.name == 'nt': cmd[0] = "soffice"
        subprocess.run(cmd, check=True)
        input_path = input_path.with_suffix(".pdf")

    if input_path.suffix.lower() == ".pdf":
        pdf = pdfium.PdfDocument(str(input_path))
        return [page.render(scale=3).to_pil() for page in pdf]

    raise ValueError(f"Unsupported file type: {ext}")

# 8. PUBLIC ENDPOINTS (No Check Needed)
# ---------------------------------------------------------
@app.post("/login")
async def login(data: LoginRequest, response: Response):
    try:
        # 1. Verify Google Token
        id_info = id_token.verify_oauth2_token(
            data.token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID
        )

        email = id_info.get("email")
        name = id_info.get("name")

        # 2. Check NSUT Domain (Make sure this is UNCOMMENTED)
        if not email.endswith("@nsut.ac.in"):
            # This raises a 402 error
            raise HTTPException(status_code=402, detail="Sign in with NSUT mail ID")

        # 3. Create Session
        session_token = secrets.token_urlsafe(32)
        sessions[session_token] = {"email": email, "name": name,"chats": []}

        response.set_cookie(
            key="auth_token",
            value=session_token,
            httponly=True,
            secure=False, 
            samesite="lax",
            max_age=604800 
        )

        return {"status": 200, "message": "Login successful", "token": session_token}

    # 👇 THIS IS THE NEW PART
    except HTTPException as e:
        # If the error is an HTTPException (like our 402), let it pass through!
        raise e 
        
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google Token")
        
    except Exception as e:
        # This only catches unexpected crashes now
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 9. PROTECTED ENDPOINTS (Using `Depends(get_current_user)`)
# ---------------------------------------------------------

@app.get("/verify")
async def verify(user: dict = Depends(get_current_user)):
    # If code reaches here, user is guaranteed to be valid
    print(f"DEBUG: User verified: {user['email']}")
    return {"status": 200, "email": user["email"], "name": user["name"], "chats": user["chats"]}

@app.get("/logout")
async def logout(response: Response, request: Request, user: dict = Depends(get_current_user)):
    token = request.cookies.get("auth_token")
    if token and token in sessions:
        del sessions[token]
    
    response.delete_cookie("auth_token")
    return {"status": 200, "message": "Logged out"}

@app.post("/train")
async def train(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    original_filename = urllib.parse.unquote(file.filename)
    _, ext = os.path.splitext(original_filename)
    # Save uploaded file
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}_{user['email']}{ext}"
    file_path = UPLOAD_DIR / safe_name
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        images = any_to_images(file_path, UPLOAD_DIR)
    except Exception as e:
        print(f"❌ CRITICAL ERROR during conversion: {e}")
        raise HTTPException(400, f"Conversion failed: {str(e)}")

    new_chunks_buffer = []

    for i, img in enumerate(images):
        current_bot = get_next_bot_client()
        try:
            base64_image = encode_image(img)
            
            # --- FULL, ELABORATIVE TRAINING PROMPT ---
            prompt_text = r"""
            You are an Advanced Technical Document Digitizer.
            Your task is to transcribe this document page into perfect, structured Markdown.

            ### CRITICAL INSTRUCTIONS:
            
            1. **MATHEMATICS & EQUATIONS (High Priority):**
               - You must detect EVERY mathematical expression, no matter how small.
               - Convert them strictly to LaTeX format.
               - For inline math, use single $: e.g., $f(x) = x^2$
               - For block math, use double $$: e.g., $$ \int_{0}^{\infty} e^{-x} dx $$
               - Do not transliterate Greek letters (write $\alpha$, not alpha).

            2. **STRUCTURAL ELEMENTS:**
               - **Tables:** Recreate tables using Markdown syntax (| Col1 | Col2 |). Do not simplify them.
               - **Headers:** Detect document hierarchy. Use # for Title, ## for Sections.
               - **Code:** If you see code snippets, wrap them in triple backticks.

            3. **DIAGRAMS & HANDWRITING:**
               - If you see a flowchart, describe it or use Mermaid.js syntax if possible.
               - If you see handwritten notes, transcribe them in *italics* with a [Handwritten] label.

            4. **OUTPUT FORMAT:**
               - Output ONLY the raw Markdown content.
               - Do not add conversational fillers like "Here is the transcription".
               - Do not wrap the output in a markdown block. Just return the text.
            """

            response = current_bot.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=2048 
            )
            
            page_text = response.choices[0].message.content
            header = f"--- Source: {file.filename}, Page: {i+1} ---\n"
            full_text = header + page_text
            
            for chunk in chunk_text(full_text):
                new_chunks_buffer.append(chunk)

            time.sleep(1.5) 

        except Exception as e:
            print(f"Error extracting page {i+1}: {e}")
            if "429" in str(e):
                time.sleep(5)

    if not new_chunks_buffer:
        raise HTTPException(500, "No text extracted.")

    embeddings = embedder.encode(new_chunks_buffer, convert_to_numpy=True).astype("float32")
    index.add(embeddings)
    chunks.extend(new_chunks_buffer)
    save_db()

    return {"status": "success", "filename": file.filename, "chunks_generated": len(new_chunks_buffer)}

@app.post("/send")
async def send_message(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None),
    user: dict = Depends(get_current_user)
):

    # --- Everything below is your original logic ---
    user.setdefault("chats", [])
    user_chats = user.get("chats", [])
    user["chats"].append({"role": "user", "content": message})

    # RAG retrieval
    if index.ntotal > 0:
        q_emb = embedder.encode([message], convert_to_numpy=True).astype("float32")
        D, I = index.search(q_emb, k=5)
        retrieved_texts = [chunks[i] for i in I[0] if i < len(chunks)]
        context_block = "\n\n".join(retrieved_texts)
    else:
        context_block = "No documents have been uploaded yet."

    # If file exists, read its content
    file_text = "No file attached"
    if file:
        contents = await file.read()
        file_text = f"Filename: {file.filename}, Size: {len(contents)} bytes"
        # later you can plug in PDF/TXT parsing here

    # --- Your system prompt unchanged ---
    system_prompt = r"""
    You are an expert Academic Teaching Assistant at NSUT.
    You are basically a RAG (Retrieval-Augmented Generation) bot. TRAINED ON NSUT DOCUMENTS ONLY.
    You have to reply as if you are talking to the student directly.
    context_block contains the data available from the documents we have in our vector db.
    if context_block is not related to the question, you can ignore it and answer on your own knowledge.
    previous_messages contains the earlier chat history with the student.
    read the previous messages to maintain context.
    give first priority to the qstn user asked and previous_messages while answering. 
    then u can use context_block to find desired data if needed.
    If the answer is not in context_block, then you can reply on the basis of your own knowledge.
    file_attached contains the file student has attached, if it is not empty then analyze that file also and answer accordingly. 
    Your goal is to answer student questions with high technical precision, using ONLY the provided Context.

    ### 1. RESPONSE STRUCTURE
    - **Direct Answer:** Start with a clear, concise answer.
    - **Step-by-Step Explanation:**
      - For **Math/Physics:** Show derivation steps using LaTeX.
      - For **Programming:** Explain logic before code.
      - For **Theory:** Break down concepts into bullet points.

    ### 2. FORMATTING RULES (STRICT)
    - **Mathematics:** Use LaTeX for ALL equations ($E=mc^2$).
    - **Diagrams:** Use `mermaid` code blocks for processes.
    - **Tables:** Use Markdown tables.
    - **Visuals:** Insert 
    [Image of X]
    tags ONLY if specific physical structures (like circuits or anatomy) are discussed.

    ### 3. GUARDRAILS
    - If the answer is NOT in the Context, state: "I cannot answer this based on the provided documents."
    - Do not hallucinate or make up facts.
    - Maintain a professional, academic tone.
    """

    user_prompt = f"""
    **Context Information:**
    {context_block}

    **Student Question:**
    {message}

    **Previous Chat History:**
    {user_chats[-8:]}

    **File Attached:**
    {file_text}
    """

    async def response_generator():
        client=get_next_bot_client()
        stream = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            stream=True
        )
        full_response = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                yield content
        user["chats"].append({"role": "assistant", "content": full_response})

    return StreamingResponse(response_generator(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)