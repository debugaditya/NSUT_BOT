import os
import shutil
import uuid
import json
import secrets
from typing import Optional
from pathlib import Path

import uvicorn
import redis
from fastapi import FastAPI, Response, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv

from worker import (
    process_file_task,
    any_to_images,
    encode_image,
    get_next_bot_client,
    get_embeddings,
    index,
)

load_dotenv()

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
GROQ_CHAT_MODEL ="meta-llama/llama-4-scout-17b-16e-instruct"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
SESSION_TTL = 604800

def get_render_user_ip(request: Request):
    real_ip = request.headers.get("True-Client-IP")
    if not real_ip:
        forwarded = request.headers.get("X-Forwarded-For")
        real_ip = forwarded.split(",")[0].strip() if forwarded else request.client.host
    return real_ip

limiter = Limiter(key_func=get_render_user_ip, storage_uri=REDIS_URL)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nsut-bot.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

class LoginRequest(BaseModel):
    token: str

def get_current_user(request: Request, response: Response):
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user_data_json = redis_client.get(token)
    if not user_data_json:
        raise HTTPException(status_code=401, detail="Session expired")
    
    response.set_cookie(
        key="auth_token", value=token, httponly=True, secure=True, 
        samesite="none", max_age=SESSION_TTL, path="/"
    )
    return json.loads(user_data_json)

@app.get("/")
def root(): return {"status": "NSUT Bot Backend is Live"}

@app.head("/health")
def health(): return {"status": "healthy"}

@app.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, data: LoginRequest, response: Response):
    try:
        id_info = id_token.verify_oauth2_token(data.token, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = id_info.get("email").lower()
        name = id_info.get("name")

        if email != "dishabarmola@gmail.com" and not email.endswith("@nsut.ac.in"):
            raise HTTPException(status_code=402, detail="Sign in with NSUT mail ID")
        
        session_token = secrets.token_urlsafe(32)
        user_data = {"email": email, "name": name, "chats": []}
        redis_client.setex(session_token, SESSION_TTL, json.dumps(user_data))
        
        response.set_cookie(
            key="auth_token", value=session_token, httponly=True, secure=True, 
            samesite="none", max_age=SESSION_TTL, path="/" 
        )
        return {"status": 200, "message": "Login successful", "token": session_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/verify")
def verify(user: dict = Depends(get_current_user)):
    return user

@app.get("/logout")
def logout(response: Response, request: Request):
    token = request.cookies.get("auth_token")
    if token: redis_client.delete(token)
    response.delete_cookie("auth_token")
    return {"status": 200, "message": "Logged out"}

@app.post("/train")
@limiter.limit("3/minute")
async def train(request: Request,file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    content_length = int(file.headers.get('content-length', 0))
    if content_length > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

    task = process_file_task.delay(str(file_path), file.filename, user['email'])
    
    return {
        "status": "success", 
        "message": "Processing started in background.", 
        "task_id": task.id
    }

@app.post("/send")
@limiter.limit("30/minute") 
async def send_message(
    request: Request,
    message: Optional[str] = Form(None), 
    file: Optional[UploadFile] = File(None),
    user: dict = Depends(get_current_user)
):
    if not message and not file:
        raise HTTPException(400, "Please provide either a message or a file.")
    msg_text = message.strip() if message else ""

    accumulated_keywords = []
    final_vision_payloads = []
    
    if file:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_BYTES:
             raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

        filename = file.filename.lower()
        temp_path = UPLOAD_DIR / f"chat_{uuid.uuid4()}_{filename}"
        
        try:
            with open(temp_path, "wb") as f: f.write(contents)

            MAX_PAGES = 3
            page_count = 0
            
            images = any_to_images(temp_path)
            
            for img in images:
                if page_count >= MAX_PAGES:
                    img.close()
                    break
                
                try:
                    b64 = encode_image(img)
                    vision_url = f"data:image/jpeg;base64,{b64}"
                    final_vision_payloads.append(vision_url)
                    
                    for _ in range(5):
                        try:
                            client = get_next_bot_client()
                            resp = client.chat.completions.create(
                                model=GROQ_CHAT_MODEL,
                                messages=[{
                                    "role": "user", 
                                    "content": [
                                        {"type": "text", "text": "List 5 key technical terms comma separated"},
                                        {"type": "image_url", "image_url": {"url": vision_url}}
                                    ]
                                }],
                                max_tokens=60
                            )
                            accumulated_keywords.append(resp.choices[0].message.content)
                            break
                        except: pass
                finally:
                    if hasattr(img, 'close'): img.close()
                page_count += 1
                
        except Exception as e:
            print(f"Chat file error: {e}")
        finally:
            if temp_path.exists(): os.remove(temp_path)

    user.setdefault("chats", [])
    user["chats"].append({"role": "user", "content": message, "document": file.filename if file else None})
    token = request.cookies.get("auth_token")
    if token: redis_client.setex(token, SESSION_TTL, json.dumps(user))

    context_block = "No documents found."
    search_query = f"{msg_text} {' '.join(accumulated_keywords)}".strip()

    if search_query:
        try:
            q_emb = get_embeddings([search_query[:2000]])[0]['values']
            results = index.query(vector=q_emb, top_k=7, include_metadata=True)
            if results['matches']:
                context_block = "\n\n".join([m['metadata']['text'] for m in results['matches']])
        except Exception as e:
            print(f"RAG Error: {e}")

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
    - Dont tell user about formatting rules or technology you are using like latex or mermaid. these are only for rendering purpose.
    - GIVE answers to user only related to the query asked. 
    - Also take care of the previous messages while answering.
    
    
    


[Image of X]



    tags ONLY if specific physical structures (like circuits or anatomy) are discussed.

    ### 3. GUARDRAILS
    - If the answer is NOT in the Context, state: "I cannot answer this based on the provided documents."
    - Do not hallucinate or make up facts.
    - Maintain a professional, academic tone.
    """

    user_prompt = f"""
    **Context:**
    {context_block}

    **Question:**
    {msg_text}

    **History:**
    {user["chats"][-6:]}
    
    **Attached Images:** {len(final_vision_payloads)}
    """

    async def response_generator():
        for _ in range(5):
            client = get_next_bot_client()
        
            messages = [{"role": "system", "content": system_prompt}]
            user_content = [{"type": "text", "text": user_prompt}]
        
            for v_url in final_vision_payloads:
                user_content.append({"type": "image_url", "image_url": {"url": v_url}})
            
            messages.append({"role": "user", "content": user_content})

            try:
                stream = client.chat.completions.create(
                    model=GROQ_CHAT_MODEL,
                    messages=messages,
                    temperature=0.3,
                    stream=True
                )
            
                full_resp = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_resp += content
                        yield content
            
                user["chats"].append({"role": "assistant", "content": full_resp})
                if token: redis_client.setex(token, SESSION_TTL, json.dumps(user))
                break
            
            except Exception as e:
                # Optional: yield nothing here to retry silently, or yield error if last attempt
                continue 

    return StreamingResponse(response_generator(), media_type="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)