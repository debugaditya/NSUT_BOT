import os
import time
import uuid
import base64
import gc
from pathlib import Path
from io import BytesIO
from typing import Generator
import ssl
from celery import Celery
import redis
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont
from markitdown import MarkItDown
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("worker")
celery_app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    broker_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    },
    redis_backend_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    }
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found.")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "nsutbot-index"
DIMENSION = 1024

existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10)

index = pc.Index(INDEX_NAME)

bot_keys = [
    os.getenv("BOT1"), os.getenv("BOT2"), os.getenv("BOT3"),
    os.getenv("BOT4"), os.getenv("BOT5")
]
valid_bot_clients = [Groq(api_key=key) for key in bot_keys if key]
if not valid_bot_clients:
    raise ValueError("No valid Groq API keys found.")

current_bot_index = 0

def get_next_bot_client():
    global current_bot_index
    selected_bot = valid_bot_clients[current_bot_index]
    current_bot_index = (current_bot_index + 1) % len(valid_bot_clients)
    return selected_bot

md_converter = MarkItDown()
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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

    try:
        result = md_converter.convert(str(input_path))
        text_content = result.text_content
        img = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            
        draw.text((40, 40), text_content[:5000], fill="black", font=font)
        yield img
    except Exception as e:
        print(f"MarkItDown error: {e}")

@celery_app.task(name="process_pdf_task")
def process_file_task(file_path_str: str, filename: str, user_email: str):
    file_path = Path(file_path_str)
    
    prompt_text = r"""
            You are an Advanced Technical Document Digitizer.
            Your task is to transcribe this document page into perfect, structured Markdown.
            Follow these rules strictly:
            
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
            base64_image = encode_image(img)
            
            for _ in range(5):
                current_bot = get_next_bot_client()
                try:
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
                    break 
                    
                except Exception:
                    time.sleep(1)
                    continue

            if hasattr(img, 'close'): img.close()
            del img
            del base64_image
            gc.collect() 
            
    except Exception as e:
        print(f"Worker Error: {e}")
    finally:
        if file_path.exists(): 
            os.remove(file_path)