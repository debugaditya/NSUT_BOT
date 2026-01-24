import os
import time
import shutil
import uuid
import base64
import urllib.parse
import secrets
from io import BytesIO
from pathlib import Path
from typing import List, Generator, Dict, Optional
import uvicorn
from markitdown import MarkItDown
import gc
import json
import redis

# Initialize MarkItDown once
md_converter = MarkItDown()

# --- Web Framework & API Imports ---
from fastapi import FastAPI, Response, HTTPException, Request, status, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- Rate Limiting Imports ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# --- Auth & AI Imports ---
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium
from groq import Groq

# --- Pinecone Import ---
from pinecone import Pinecone, ServerlessSpec


# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
load_dotenv()

# --- CONFIGURATION CONSTANTS ---
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- REDIS CLIENT SETUP (FOR SESSIONS) ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
SESSION_TTL = 604800  # 7 Days in seconds

# --- Rate Limiting Strategy for Render ---
def get_render_user_ip(request: Request):
    real_ip = request.headers.get("True-Client-IP")
    if not real_ip:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            real_ip = forwarded.split(",")[0].strip()
        else:
            real_ip = request.client.host
    return real_ip

# Initialize Limiter
limiter = Limiter(
    key_func=get_render_user_ip,
    storage_uri=os.getenv("REDIS_URL", "memory://")
)

app = FastAPI()

# Attach Limiter to App
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Credentials & URLs
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found in environment variables.")

# Models
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_CHAT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL = "llama-text-embed-v2"
DIMENSION = 1024 

# Directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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


# 3. DATABASE INITIALIZATION (PINECONE)
# ---------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "nsutbot-index"

existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10)

index = pc.Index(INDEX_NAME)
print(f"✅ Connected to Pinecone index: {INDEX_NAME}")

# 4. MIDDLEWARE
# ---------------------------------------------------------
origins = [
    "https://nsut-bot.vercel.app",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. SECURITY DEPENDENCY
# ---------------------------------------------------------
def get_current_user(request: Request, response: Response):
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    
    user_data_json = redis_client.get(token)
    if not user_data_json:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    
    response.set_cookie(
        key="auth_token", value=token, httponly=True, secure=True, 
        samesite="none", max_age=604800, path="/"
    )
    return json.loads(user_data_json)

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

def encode_image(pil_image):
    buffered = BytesIO()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_embeddings(texts):
    return pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=texts,
        parameters={"input_type": "passage"}
    )

def any_to_images(input_path: Path) -> Generator[Image.Image, None, None]:
    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    
    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        yield Image.open(input_path)
        return

    # PDF Processing
    if ext == ".pdf":
        pdf = pdfium.PdfDocument(str(input_path))
        try:
            for page in pdf:
                pil_image = page.render(scale=2).to_pil()
                yield pil_image
                del pil_image 
        finally:
            pdf.close()
        return

    # Office/Text Processing
    if ext in [".docx", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md"]:
        try:
            result = md_converter.convert(str(input_path))
            text_content = result.text_content
            # Convert text to a simple image
            img = Image.new("RGB", (1200, 1600), "white")
            draw = ImageDraw.Draw(img)
            # Use default font or load one if available. Default is machine dependent but works for basic text.
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
                
            draw.text((40, 40), text_content[:5000], fill="black", font=font)
            yield img
        except Exception as e:
            print(f"❌ MarkItDown error: {e}")

def process_file_in_background(file_path: Path, filename: str, user_email: str):
    # (Background Training Logic - Unchanged)
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
    try:
        images = any_to_images(file_path)
        for i, img in enumerate(images):
            current_bot = get_next_bot_client()
            try:
                base64_image = encode_image(img)
                response = current_bot.chat.completions.create(
                    model=GROQ_VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }],
                    temperature=0.1
                )
                page_text = response.choices[0].message.content
                chunks = list(chunk_text(page_text))
                if chunks:
                    emb_res = get_embeddings(chunks)
                    vectors = []
                    for chunk, emb in zip(chunks, emb_res):
                        vectors.append({
                            "id": str(uuid.uuid4()),
                            "values": emb['values'],
                            "metadata": {"text": chunk, "filename": filename, "user_email": user_email}
                        })
                    index.upsert(vectors=vectors)
            finally:
                if hasattr(img, 'close'): img.close()
                del img
                del base64_image
                gc.collect() 
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
    finally:
        if file_path.exists(): os.remove(file_path)

# 8. PUBLIC ENDPOINTS
# ---------------------------------------------------------
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "NSUT Bot Backend is Live"}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "healthy"}

@app.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, data: LoginRequest, response: Response):
    try:
        id_info = id_token.verify_oauth2_token(data.token, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = id_info.get("email")
        name = id_info.get("name")
        if email!="dishabarmola@gmail.com" and not email.endswith("@nsut.ac.in"):
            raise HTTPException(status_code=402, detail="Sign in with NSUT mail ID")
        
        session_token = secrets.token_urlsafe(32)
        user_data = {"email": email, "name": name, "chats": []}
        redis_client.setex(session_token, SESSION_TTL, json.dumps(user_data))
        
        response.set_cookie(
            key="auth_token", value=session_token, httponly=True, secure=True, 
            samesite="none", max_age=604800, path="/" 
        )
        return {"status": 200, "message": "Login successful", "token": session_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 9. PROTECTED ENDPOINTS
# ---------------------------------------------------------

@app.get("/verify")
async def verify(user: dict = Depends(get_current_user)):
    return {"status": 200, "email": user["email"], "name": user["name"], "chats": user["chats"]}

@app.get("/logout")
async def logout(response: Response, request: Request, user: dict = Depends(get_current_user)):
    token = request.cookies.get("auth_token")
    if token:
        redis_client.delete(token)
    response.delete_cookie("auth_token")
    return {"status": 200, "message": "Logged out"}

@app.post("/train")
@limiter.limit("10/minute")
async def train(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    original_filename = urllib.parse.unquote(file.filename)
    _, ext = os.path.splitext(original_filename)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}_{user['email']}{ext}"
    file_path = UPLOAD_DIR / safe_name
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            os.remove(file_path)
            raise HTTPException(status_code=413, detail="File too large.")
    except Exception as e:
        if file_path.exists(): os.remove(file_path)
        raise e

    background_tasks.add_task(process_file_in_background, file_path, file.filename, user['email'])
    return {"status": "success", "message": "Processing in background."}

# =========================================================
#  UPDATED /SEND ENDPOINT (VISION-BASED KEYWORDS & CONTEXT)
# =========================================================
@app.post("/send")
@limiter.limit("60/minute") 
async def send_message(
    request: Request,
    message: Optional[str] = Form(None), 
    file: Optional[UploadFile] = File(None),
    user: dict = Depends(get_current_user)
):
    # 1. Validation
    if not message and not file:
        raise HTTPException(status_code=400, detail="Please provide either a message or a file.")
    msg_text = message.strip() if message else ""

    # 2. FILE PROCESSING (Universal Vision Conversion)
    # ---------------------------------------------------------
    accumulated_keywords = []  # To store keywords from all pages
    final_vision_payloads = [] # To store image URLs for the final prompt
    
    if file:
        # Check size immediately
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_BYTES:
             raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

        filename = file.filename.lower()
        temp_path = UPLOAD_DIR / f"temp_chat_{uuid.uuid4()}_{filename}"
        
        try:
            # Save temp file for processing
            with open(temp_path, "wb") as f:
                f.write(contents)

            # Process using the Vision Generator
            # NOTE: We limit to first 3 pages to prevent request timeout during chat
            MAX_PAGES_TO_SCAN = 3 
            page_count = 0
            
            image_generator = any_to_images(temp_path)
            
            for img in image_generator:
                if page_count >= MAX_PAGES_TO_SCAN:
                    img.close()
                    break # Stop processing to save time
                
                try:
                    # 1. Encode Image
                    base64_image = encode_image(img)
                    vision_url = f"data:image/jpeg;base64,{base64_image}"
                    
                    # 2. Add to Final Context (So AI sees it while answering)
                    final_vision_payloads.append(vision_url)
                    
                    # 3. Extract Keywords (For RAG Search)
                    # We try 2 times max per page to get keywords
                    for _ in range(2): 
                        try:
                            scout_client = get_next_bot_client()
                            scout_response = scout_client.chat.completions.create(
                                model=GROQ_CHAT_MODEL,
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Analyze this page. Identify the main scientific topics, formulas, and key terms. Return only a comma-separated list of keywords."},
                                        {"type": "image_url", "image_url": {"url": vision_url}}
                                    ]
                                }],
                                max_tokens=60  # Short output for speed
                            )
                            keywords = scout_response.choices[0].message.content
                            accumulated_keywords.append(keywords)
                            break # Success, move to next page
                        except Exception as e:
                            print(f"⚠️ Keyword extraction failed on page {page_count}: {e}")
                            continue

                finally:
                    # RAM OPTIMIZATION: Close resources immediately
                    if hasattr(img, 'close'): img.close()
                    del img
                    del base64_image
                    gc.collect() # Force cleanup
                
                page_count += 1
                
        except Exception as e:
            print(f"Error processing file for chat: {e}")
        finally:
            # Cleanup temp file
            if temp_path.exists(): os.remove(temp_path)

    # 3. Update Chat History & Redis
    user.setdefault("chats", [])
    user_chats = user.get("chats", [])
    user["chats"].append({"role": "user", "content": message, "document": file.filename if file else None})
    
    token = request.cookies.get("auth_token")
    if token:
        redis_client.setex(token, SESSION_TTL, json.dumps(user))

    # 4. RAG RETRIEVAL (Intelligent Search)
    # ---------------------------------------------------------
    context_block = "No documents found."
    
    # Search Query = User Text + All Keywords extracted from images
    image_search_keywords_str = " ".join(accumulated_keywords)
    search_query = f"{msg_text} {image_search_keywords_str}"

    if search_query.strip():
        try:
            # Limit query length to prevent embedding errors
            query_response = get_embeddings([search_query[:2000]]) 
            q_emb = query_response[0]['values']
            
            search_results = index.query(vector=q_emb, top_k=7, include_metadata=True)
            if search_results['matches']:
                retrieved_texts = [match['metadata']['text'] for match in search_results['matches']]
                context_block = "\n\n".join(retrieved_texts)
        except Exception as e:
            print(f"Error during RAG retrieval: {e}")

    # 5. PROMPTS
    # ---------------------------------------------------------
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
    {msg_text}

    **Previous Chat History:**
    {user_chats[-8:]}

    **Attached File Analysis:**
    I have attached {len(final_vision_payloads)} images representing the document the user uploaded.
    Use these images to answer the question if relevant.
    """

    # 6. RESPONSE GENERATOR
    async def response_generator():
        max_retries = len(valid_bot_clients)
        
        for attempt in range(max_retries):
            try:
                client = get_next_bot_client()
                
                # --- CONSTRUCT PAYLOAD ---
                messages_payload = [{"role": "system", "content": system_prompt}]
                
                # Base Content (Text)
                user_content_list = [{"type": "text", "text": user_prompt}]
                
                # Append Vision Payloads (The actual images)
                for vision_url in final_vision_payloads:
                    user_content_list.append({
                        "type": "image_url", 
                        "image_url": {"url": vision_url}
                    })

                messages_payload.append({
                    "role": "user",
                    "content": user_content_list
                })

                stream = client.chat.completions.create(
                    model=GROQ_CHAT_MODEL, 
                    messages=messages_payload,
                    temperature=0.3,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield content
                
                # Write back assistant msg
                user["chats"].append({"role": "assistant", "content": full_response})
                if token:
                    redis_client.setex(token, SESSION_TTL, json.dumps(user))
                return 

            except Exception as e:
                print(f"⚠️ Key failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    yield f"\n\n[System Error] All AI models are currently unavailable. Please contact support."

    return StreamingResponse(response_generator(), media_type="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)