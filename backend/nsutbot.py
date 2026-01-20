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

# Initialize MarkItDown once
md_converter = MarkItDown()

# --- Web Framework & API Imports ---
from fastapi import FastAPI, Response, HTTPException, Request, status, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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

app = FastAPI()

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
DIMENSION = 1024  # Dimension for Llama 2 embeddings

# Directories
# (We no longer need DB_DIR for local vectors, but we keep UPLOAD_DIR)
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

# Check if index exists, if not create it
existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    # Wait a moment for index to be initialized
    time.sleep(10)

# Connect to the index
index = pc.Index(INDEX_NAME)
print(f"✅ Connected to Pinecone index: {INDEX_NAME}")

# In-memory session store
sessions = {}

# 4. MIDDLEWARE
# ---------------------------------------------------------
origins = [
    "https://nsut-bot.vercel.app",  # Your live Vercel frontend
    "http://localhost:5173",       # Local development
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. SECURITY DEPENDENCY (THE GATEKEEPER)
# ---------------------------------------------------------
def get_current_user(request: Request,response: Response):
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
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        secure=True, 
        samesite="none",
        max_age=604800,
        path="/"
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
        try:
            # Direct conversion to Markdown (No PDF or LibreOffice needed!)
            result = md_converter.convert(str(input_path))
            text_content = result.text_content
            
            # Create a high-resolution white canvas to "print" the text for the Vision model
            img = Image.new("RGB", (1200, 1600), "white")
            draw = ImageDraw.Draw(img)
            
            # Use a standard Linux font (DejaVu) or fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw the first 5000 characters of the docx onto the image
            draw.text((40, 40), text_content[:5000], fill="black", font=font)
            return [img]
            
        except Exception as e:
            print(f"❌ MarkItDown error on {input_path.name}: {e}")
            return []
    if ext in [".pptx", ".ppt", ".xlsx", ".xls"]:
        try:
            # MarkItDown handles these formats directly in Python
            result = md_converter.convert(str(input_path))
            text_content = result.text_content
            
            # Create a "virtual page" for your Vision AI to maintain your OCR pipeline
            img = Image.new("RGB", (1200, 1600), "white")
            draw = ImageDraw.Draw(img)
            
            try:
                # Use a standard Linux font path
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font = ImageFont.load_default()
            
            # Draw extracted text (Slides/Sheets) onto the image
            draw.text((40, 40), text_content[:5000], fill="black", font=font)
            return [img]

        except Exception as e:
            print(f"❌ Error processing Office file {input_path.name}: {e}")
            return []

    if input_path.suffix.lower() == ".pdf":
        pdf = pdfium.PdfDocument(str(input_path))
        return [page.render(scale=3).to_pil() for page in pdf]

    raise ValueError(f"Unsupported file type: {ext}")
def process_file_in_background(file_path: Path, filename: str, user_email: str):
    """
    Runs in the background AFTER the user gets a response.
    Handles OCR, Chunking, Embedding, and Pinecone Upsert.
    """
    print(f"🔄 Background Task Started: Processing {filename} for {user_email}...")
    
    try:
        # 1. Convert PDF/Doc to Images
        try:
            images = any_to_images(file_path, UPLOAD_DIR)
        except Exception as e:
            print(f"❌ Error converting file {filename}: {e}")
            return

        new_chunks_buffer = []

        # 2. Extract Text from Each Page (OCR)
        for i, img in enumerate(images):
            current_bot = get_next_bot_client()
            try:
                base64_image = encode_image(img)
                
                # --- YOUR ORIGINAL PROMPT ---
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
                header = f"--- Source: {filename}, Page: {i+1} ---\n"
                full_text = header + page_text
                
                for chunk in chunk_text(full_text):
                    new_chunks_buffer.append(chunk)

                # Rate limit protection
                time.sleep(1.5)
                img.close() 

            except Exception as e:
                print(f"⚠️ Error extracting page {i+1} of {filename}: {e}")
                if "429" in str(e):
                    time.sleep(5)
                    continue

        if not new_chunks_buffer:
            print(f"❌ Aborted: No text extracted from {filename}")
            return

        # 3. Embed & Upsert to Pinecone
        print(f"⚡ Generating embeddings for {len(new_chunks_buffer)} chunks...")
        embedding_response = get_embeddings(new_chunks_buffer)
        embeddings = [item['values'] for item in embedding_response]

        vectors_to_upsert = []
        for chunk, emb in zip(new_chunks_buffer, embeddings):
            vector_id = str(uuid.uuid4())
            metadata = {
                "text": chunk,
                "filename": filename,
                "user_email": user_email
            }
            vectors_to_upsert.append({
                "id": vector_id,
                "values": emb,
                "metadata": metadata
            })

        # Batch Upsert (100 at a time)
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i : i + batch_size]
            index.upsert(vectors=batch)

        print(f"✅ Success: {filename} is now live in the knowledge base!")

    except Exception as e:
        print(f"❌ CRITICAL BACKGROUND ERROR: {e}")
    
    
    if file_path.exists():
        os.remove(file_path)

# 8. PUBLIC ENDPOINTS (No Check Needed)
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "NSUT Bot Backend is Live"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
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
            secure=True, 
            samesite="none",
            max_age=604800,
            path="/" 
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
async def train(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    user: dict = Depends(get_current_user)
):
    original_filename = urllib.parse.unquote(file.filename)
    _, ext = os.path.splitext(original_filename)
    
    # 1. Save uploaded file immediately
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}_{user['email']}{ext}"
    file_path = UPLOAD_DIR / safe_name
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Add the heavy task to the background queue
    background_tasks.add_task(process_file_in_background, file_path, file.filename, user['email'])

    # 3. Return "Success" immediately so user can close window
    return {
        "status": "success",
        "message": "Upload successful! We are processing the file in the background. You can close this window now.",
        "filename": file.filename
    }

@app.post("/send")
async def send_message(
    message: Optional[str] = Form(None), # 👈 Changed to Optional
    file: Optional[UploadFile] = File(None),
    user: dict = Depends(get_current_user)
):
    if not message and not file:
        raise HTTPException(status_code=400, detail="Please provide either a message or a file.")
    if message is None:
        message = ""

    # --- Everything below is your original logic ---
    user.setdefault("chats", [])
    user_chats = user.get("chats", [])
    user["chats"].append({"role": "user", "content": message, "document": file.filename if file else None})

    # RAG retrieval (PINECONE)
    context_block = "No documents have been uploaded yet."
    
    # Generate query embedding
    query_response = get_embeddings([message])
    q_emb = query_response[0]['values']
    
    # Query Pinecone
    try:
        search_results = index.query(
            vector=q_emb,
            top_k=10,
            include_metadata=True
        )
        
        if search_results['matches']:
            retrieved_texts = [match['metadata']['text'] for match in search_results['matches']]
            context_block = "\n\n".join(retrieved_texts)
            
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        # Fallback to no context if DB fails

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
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
