from pathlib import Path
import uuid

import chromadb
from fastapi import APIRouter, UploadFile, File, HTTPException
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# ==================================================
# CONFIGURATION
# ==================================================

UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ALLOWED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx"
}


# ==================================================
# EMBEDDING MODEL
# ==================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# CHROMADB
# ==================================================

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="project_documents"
)


# ==================================================
# ROUTER
# ==================================================

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


# ==================================================
# TEXT FILE
# ==================================================

def extract_txt(content: bytes) -> str:

    try:

        return content.decode(
            "utf-8"
        )

    except UnicodeDecodeError:

        raise HTTPException(
            status_code=400,
            detail="TXT file must be UTF-8 encoded."
        )


# ==================================================
# PDF FILE
# ==================================================

def extract_pdf(file_path: Path) -> str:

    try:

        reader = PdfReader(
            str(file_path)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:

                pages.append(text)

        return "\n".join(pages)

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read PDF: {str(e)}"
        )


# ==================================================
# DOCX FILE
# ==================================================

def extract_docx(file_path: Path) -> str:

    try:

        document = Document(
            str(file_path)
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                paragraphs.append(text)

        return "\n".join(paragraphs)

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read DOCX: {str(e)}"
        )


# ==================================================
# CHUNKING
# ==================================================

def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(
                chunk
            )

        # Move forward but keep overlap
        start += chunk_size - overlap


    return chunks


# ==================================================
# UPLOAD DOCUMENT
# ==================================================

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # ----------------------------------------------
    # Validate filename
    # ----------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )


    # ----------------------------------------------
    # Get extension
    # ----------------------------------------------

    safe_filename = Path(
        file.filename
    ).name

    extension = Path(
        safe_filename
    ).suffix.lower()


    # ----------------------------------------------
    # Validate extension
    # ----------------------------------------------

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported files: TXT, PDF and DOCX."
            )
        )


    # ----------------------------------------------
    # Read uploaded file
    # ----------------------------------------------

    content = await file.read()


    if not content:

        raise HTTPException(
            status_code=400,
            detail="The uploaded document is empty."
        )


    # ----------------------------------------------
    # Save original file
    # ----------------------------------------------

    destination = (
        UPLOAD_DIR /
        safe_filename
    )

    destination.write_bytes(
        content
    )


    # ----------------------------------------------
    # Extract text
    # ----------------------------------------------

    if extension == ".txt":

        text = extract_txt(
            content
        )

    elif extension == ".pdf":

        text = extract_pdf(
            destination
        )

    elif extension == ".docx":

        text = extract_docx(
            destination
        )

    else:

        raise HTTPException(
            status_code=400,
            detail="Unsupported document type."
        )


    # ----------------------------------------------
    # Validate extracted text
    # ----------------------------------------------

    if not text.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found "
                "in the uploaded document."
            )
        )


    # ----------------------------------------------
    # Create chunks
    # ----------------------------------------------

    chunks = create_chunks(
        text
    )


    if not chunks:

        raise HTTPException(
            status_code=400,
            detail="No readable chunks were created."
        )


    # ----------------------------------------------
    # Generate embeddings
    # ----------------------------------------------

    embeddings = embedding_model.encode(
        chunks
    ).tolist()


    # ----------------------------------------------
    # Create IDs + metadata
    # ----------------------------------------------

    ids = []

    metadatas = []


    for index in range(
        len(chunks)
    ):

        ids.append(
            f"upload_{uuid.uuid4().hex}"
        )

        metadatas.append({

            "source":
                safe_filename,

            "category":
                "Uploads",

            "file_type":
                extension.replace(
                    ".",
                    ""
                ),

            "chunk_index":
                index

        })


    # ----------------------------------------------
    # Store in ChromaDB
    # ----------------------------------------------

    collection.add(

        documents=chunks,

        embeddings=embeddings,

        ids=ids,

        metadatas=metadatas

    )


    # ----------------------------------------------
    # Response
    # ----------------------------------------------

    return {

        "message":
            "Document uploaded and indexed successfully.",

        "filename":
            safe_filename,

        "file_type":
            extension.replace(
                ".",
                ""
            ),

        "chunks_stored":
            len(chunks),

        "category":
            "Uploads"

    }