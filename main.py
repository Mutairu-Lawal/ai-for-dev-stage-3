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
import re
from typing import List, Optional, Tuple


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

HF_TOKEN = os.getenv('HF_TOKEN', '')
EMBED_MODEL_NAME = os.getenv('EMBED_MODEL_NAME', 'all-MiniLM-L6-v2')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'gemini-2.5-flash')
CHROMA_DB_HOST = os.getenv('CHROMA_DB_HOST', '')
RAG_DATA_DIR = os.getenv('RAG_DATA_DIR', './data')
CHUNK_SIZE = int(os.getenv('CHUNK_LENGTH', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
PORT = int(os.getenv('PORT', '8080'))


app = FastAPI(title="RAG System", version="1.0",
              description="Retrieval-Augmented Generation System API with Semantic Chunking")
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])


# LLM model
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel(LLM_MODEL_NAME)

# ChromaDB client
chroma_client = chromadb.Client()
try:
    collection = chroma_client.get_collection(name="my_rag_collection")
except:
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


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Semantically chunk text by sentences while respecting size and overlap.
    Splits on sentence boundaries (.!?) to preserve semantic meaning.
    """
    # Split into sentences while preserving punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_chunk = ""
    current_start = text.find(sentences[0])

    for i, sentence in enumerate(sentences):
        if not current_chunk:
            current_chunk = sentence
            current_start = text.find(sentence, current_start if i == 0 else 0)
        elif len(current_chunk) + len(sentence) + 1 <= size:
            current_chunk += " " + sentence
        else:
            # Save current chunk
            chunk = current_chunk.strip()
            if chunk:
                current_end = current_start + len(chunk)
                chunks.append((chunk, (current_start, current_end)))

            # Start new chunk with overlap
            overlap_text = current_chunk[max(0, len(current_chunk) - overlap):]
            current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            current_start = text.find(sentence, current_start)

    # Add last chunk
    if current_chunk.strip():
        chunk = current_chunk.strip()
        current_end = current_start + len(chunk)
        chunks.append((chunk, (current_start, current_end)))

    return chunks


# ---- Endpoints ------- #
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), context: Optional[str] = Form(None)):
    try:
        if context is None:
            context = f"ctx-{uuid.uuid4().hex[:8]}"

        ctx_dir = os.path.join(RAG_DATA_DIR, context)
        os.makedirs(ctx_dir, exist_ok=True)
        file_dir = os.path.join(ctx_dir, "files")
        os.makedirs(file_dir, exist_ok=True)

        metadata_path = os.path.join(ctx_dir, "metadata.json")
        metadata = []
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)

        new_vectors = []

        for f in files:
            content = await f.read()
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

        # upsert into chromadb
        collection.add(documents=[v[2]["text"] for v in new_vectors],
                       ids=[v[0] for v in new_vectors],
                       metadatas=[v[2] for v in new_vectors],
                       embeddings=[v[1] for v in new_vectors])

        # Save the metadata.
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {"context": context, "chunks": len(new_vectors)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading files: {str(e)}")


@app.post("/prompt")
async def prompt(context: str = Form(...), query: str = Form(...)):
    try:
        # Embed query
        qvec = embed_model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[qvec],
            n_results=5,
            include=["documents", "metadatas"]
        )

        # Chroma returns lists inside lists
        if not results["documents"] or not results["documents"][0]:
            raise HTTPException(
                status_code=404,
                detail="No matching documents found"
            )

        # Extract retrieved text
        retrieved = results["documents"][0]

        context_block = "\n".join(retrieved)

        prompt_text = f"""
Context:
{context_block}

Question: {query}

Based on the context provided above, generate a succinct answer.
"""

        response = llm_model.generate_content(prompt_text)

        return {
            "answer": response.text,
            "context": retrieved
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/rechunk")
async def rechunk(payload: dict):
    try:
        context = payload.get("context")
        if not context:
            raise HTTPException(
                status_code=400,
                detail="Missing 'context' in payload"
            )

        new_size = int(payload.get("new_chunk_size", CHUNK_SIZE))
        new_overlap = int(payload.get("new_chunk_overlap", CHUNK_OVERLAP))

        results = collection.get(
            where={"context": context},
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            return {"message": "No documents found for this context"}

        ctx_dir = os.path.join(RAG_DATA_DIR, context)
        file_dir = os.path.join(ctx_dir, "files")

        if not os.path.exists(file_dir):
            raise HTTPException(
                status_code=404,
                detail="Context files not found"
            )

        # Delete old chunks
        collection.delete(ids=results["ids"])

        new_ids, new_docs, new_metas, new_embeds = [], [], [], []

        for filename in os.listdir(file_dir):
            file_path = os.path.join(file_dir, filename)

            with open(file_path, "rb") as f:
                content = f.read()

            text = extract_text(filename, content)
            chunks = chunk_text(text, size=new_size, overlap=new_overlap)

            for chunk, (s, e) in chunks:
                cid = uuid.uuid4().hex
                new_ids.append(cid)
                new_docs.append(chunk)
                new_metas.append({
                    "context": context,
                    "filename": filename,
                    "offset_start": s,
                    "offset_end": e
                })
                new_embeds.append(embed_model.encode(chunk).tolist())

        if new_ids:
            collection.upsert(
                ids=new_ids,
                documents=new_docs,
                metadatas=new_metas,
                embeddings=new_embeds
            )

        metadata_path = os.path.join(ctx_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(new_metas, f, indent=2)

        return {
            "message": "Rechunking completed",
            "context": context,
            "new_chunks": len(new_ids),
            "new_chunk_size": new_size,
            "new_chunk_overlap": new_overlap
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during rechunking: {str(e)}"
        )


@app.get("/health", status_code=200)
async def health_check():
    return {
        "status": "ok",
        "message": "App is live"
    }
