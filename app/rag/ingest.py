from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Root folder containing all knowledge sources
DATA_PATH = Path("data")

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="project_documents"
)


def load_documents():
    """
    Read every .txt file inside data/
    and all of its subfolders.
    """

    documents = []

    for file_path in DATA_PATH.rglob("*.txt"):

        # Ignore files inside the memory folder
        if "memory" in file_path.parts:
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        if not text:
            continue

        documents.append({
            "name": file_path.name,
            "category": file_path.parent.name,
            "path": str(file_path),
            "text": text
        })

    return documents


def create_chunks(text):
    """
    Create chunks based on document sections.

    Sections are separated by headings written
    in uppercase, such as JUPITER, SATURN, etc.
    """

    lines = text.splitlines()

    chunks = []
    current_chunk = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Detect uppercase section headings
        if (
            line.isupper()
            and len(line.split()) <= 8
        ):

            # Store previous section
            if current_chunk:
                chunks.append(
                    " ".join(current_chunk)
                )

            current_chunk = [line]

        else:
            current_chunk.append(line)

    # Store final section
    if current_chunk:
        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


def store_chunks(chunks, source, category):
    """
    Create embeddings and store chunks
    with source metadata in ChromaDB.
    """

    if not chunks:
        return

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [
        f"{category}_{source}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": source,
            "category": category
        }
        for _ in chunks
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


if __name__ == "__main__":

    documents = load_documents()

    total_chunks = 0

    print("\nDocuments found:\n")

    for document in documents:

        print(
            f"- {document['category']}/"
            f"{document['name']}"
        )

        chunks = create_chunks(
            document["text"]
        )

        store_chunks(
            chunks,
            document["name"],
            document["category"]
        )

        total_chunks += len(chunks)

    print(
        f"\nStored {total_chunks} chunks "
        f"from {len(documents)} documents."
    )