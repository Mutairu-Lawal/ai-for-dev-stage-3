# Python RAG template (FastAPI)
# Endpoints: /upload, /prompt, /rechunk

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from io import BytesIO
import json
from typing import List, Optional


# File parsing
try:
    import PyPDF2
except:
    PyPDF2 = None

try:
    import docx
except:
    docx = None


# Load environment variables
load_dotenv()


HF_TOKEN = os.getenv('HF_TOKEN')
EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', 'all-MiniLM-L6-v2')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'gpt-3.5-turbo')
CHROMA_DB_HOST = os.getenv('CHROMA_DB_HOST')
RAG_DATA_DIR = os.getenv('RAG_DATA_DIR', './data')
CHUNK_SIZE = int(os.getenv('CHUNK_LENGTH', '500'))
CHUNK_OVERLAP = 100
PORT = int(os.getenv('PORT', '8080'))


app = FastAPI(title="RAG System", version="1.0",
              description="Retrieval-Augmented Generation System API")
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])


# LLM model
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel(LLM_MODEL_NAME)

#  ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_rag_collection")

# Embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        try:
            return content.decode("utf-8")
        except:
            return content.decode("latin-1", errors="ignore")

    if lower.endswith(".pdf"):
        if PyPDF2 is None:
            raise HTTPException(500, "PyPDF2 not installed")

        reader = PyPDF2.PdfReader(BytesIO(content))
        txt = []
        for p in reader.pages:
            try:
                txt.append(p.extract_text() or "")
            except:
                txt.append("")

        return "\n".join(txt)

    if lower.endswith(".docx"):
        if docx is None:
            raise HTTPException(500, "python-docx not installed")

        d = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in d.paragraphs])

    try:
        return content.decode("utf-8")
    except:
        return content.decode("latin-1", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, (start, end)))

        if end == length:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


# ---- Endpoints ------- #
@app.post("/upload")
def upload_files(files: List[UploadFile] = File(...), context: Optional[str] = Form(None)):
    if context is None:
        context = f"ctx-{uuid.uuid4().hex[:8]}"

    ctx_dir = os.path.join(RAG_DATA_DIR, context)
    os.makedirs(ctx_dir, exist_ok=True)
    file_dir = os.path.join(ctx_dir, "files")
    os.makedirs(file_dir, exist_ok=True)

    metadata_path = os.path.join(ctx_dir, "metadata.json")
    metadata = []
    if os.path.exists(metadata_path):
        metadata = json.load(open(metadata_path))

    new_vectors = []

    for f in files:
        content = f.file.read()
        text = extract_text(f.filename, content)
        chunks = chunk_text(text)

        # save file
        dest = os.path.join(file_dir, f.filename)
        with open(dest, "wb") as out:
            out.write(content)

        # process chunks
        for chunk, (s, e) in chunks:
            vec = embed_model.encode(chunk).tolist()
            cid = uuid.uuid4().hex
            meta = {
                "id": cid,
                "context": context,
                "filename": f.filename,
                "offset_start": s,
                "offset_end": e,
                "text": chunk,
            }
            new_vectors.append((cid, vec, meta))
            metadata.append(meta)

    # Upsert into chromadb
    collection.upsert(documents=[v[2]["text"] for v in new_vectors],
                      ids=[v[0] for v in new_vectors],
                      metadatas=[v[2] for v in new_vectors],
                      embeddings=[v[1] for v in new_vectors])

    # Save the metadata.
    json.dump(metadata, open(metadata_path, "w"), indent=2)

    return {"context": context, "chunks": len(new_vectors)}


@app.post("/chat")
def chat(context: str = Form(...), query: str = Form(...)):
    # embed query
    qvec = embed_model.encode(query).tolist()
    results = collection.query(vector=qvec, top_k=5,
                               include_metadata=True, filter={"context": context})

    retrieved = [m["metadata"]["text"] for m in results["matches"]]
    context_block = "\n".join(retrieved)

    # With LLM
    prompt = f"""
Context:
    {context_block}

    Question: {query}
    
    Based on the context provided above, generate a succint answer to the query above.
"""
    response = llm_model.generate_content(prompt)

    return {"answer": response.text, "context": retrieved}


@app.post("/rechunk")
async def rechunk(payload: dict):
    # TODO: implement rechunk handling
    return {"message": "Rechunk endpoint not yet implemented"}
