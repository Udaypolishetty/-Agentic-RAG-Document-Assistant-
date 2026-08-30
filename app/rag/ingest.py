from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENT_PATH = Path("data/documents/project.txt")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="project_documents"
)


def load_document():
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        return file.read()


def create_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def store_chunks(chunks):
    embeddings = embedding_model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )


if __name__ == "__main__":
    document = load_document()
    chunks = create_chunks(document)

    store_chunks(chunks)

    print(f"Stored {len(chunks)} chunks in ChromaDB.")