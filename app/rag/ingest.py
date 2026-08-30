from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("data")


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="project_documents"
)


# --------------------------------------------------
# Read TXT
# --------------------------------------------------

def read_txt(path: Path) -> str:

    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


# --------------------------------------------------
# Read PDF
# --------------------------------------------------

def read_pdf(path: Path) -> str:

    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


# --------------------------------------------------
# Read DOCX
# --------------------------------------------------

def read_docx(path: Path) -> str:

    document = Document(str(path))

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


# --------------------------------------------------
# Read document based on extension
# --------------------------------------------------

def read_document(path: Path) -> str:

    extension = path.suffix.lower()

    if extension == ".txt":

        return read_txt(path)

    elif extension == ".pdf":

        return read_pdf(path)

    elif extension == ".docx":

        return read_docx(path)

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )


# --------------------------------------------------
# Category detection
# --------------------------------------------------

def detect_category(path: Path) -> str:

    parts = [
        part.lower()
        for part in path.parts
    ]

    if "companies" in parts:
        return "Company"

    if "jobs" in parts:
        return "Jobs"

    if "projects" in parts:
        return "Projects"

    if "personal" in parts:
        return "Personal"

    if "documents" in parts:
        return "Documents"

    if "memory" in parts:
        return "Memory"

    return "General"


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
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

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# Find all supported documents
# --------------------------------------------------

def find_documents():

    documents = []

    for path in DATA_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in {
            ".txt",
            ".pdf",
            ".docx"
        }:

            documents.append(path)

    return documents


# --------------------------------------------------
# Store documents
# --------------------------------------------------

def store_document(
    path: Path,
    document_index: int
):

    text = read_document(path)

    if not text.strip():

        print(
            f"Skipping empty document: {path}"
        )

        return 0

    chunks = create_chunks(text)

    if not chunks:

        return 0

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    category = detect_category(path)

    ids = []

    metadatas = []

    for chunk_index in range(
        len(chunks)
    ):

        chunk_id = (
            f"document_{document_index}_"
            f"chunk_{chunk_index}"
        )

        ids.append(chunk_id)

        metadatas.append({

            "source": path.name,

            "path": str(path),

            "category": category,

            "chunk_index": chunk_index
        })


    collection.add(

        documents=chunks,

        embeddings=embeddings,

        ids=ids,

        metadatas=metadatas
    )

    return len(chunks)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    documents = find_documents()

    print(
        f"\nFound {len(documents)} documents.\n"
    )

    total_chunks = 0

    for index, document in enumerate(
        documents
    ):

        print(
            f"Processing: {document}"
        )

        chunks_stored = store_document(
            document,
            index
        )

        total_chunks += chunks_stored

        print(
            f"  → {chunks_stored} chunks stored"
        )


    print(
        "\n--------------------------------"
    )

    print(
        f"Documents processed: {len(documents)}"
    )

    print(
        f"Total chunks stored: {total_chunks}"
    )

    print(
        "--------------------------------"
    )