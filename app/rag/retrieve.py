import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Embedding Model
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

collection = client.get_collection(
    name="project_documents"
)


# --------------------------------------------------
# Search
# --------------------------------------------------

def search(query, n_results=3):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted_results = []

    for i, document in enumerate(documents):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else None
        )

        formatted_results.append({

            "content": document,

            "source": metadata.get(
                "source",
                "Unknown"
            ),

            "category": metadata.get(
                "category",
                "Unknown"
            ),

            "distance": distance
        })

    return formatted_results


# --------------------------------------------------
# Command Line Test
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "Ask a question: "
    )

    results = search(
        question,
        n_results=4
    )

    print(
        "\nRelevant information:\n"
    )

    for result in results:

        print(
            f"[{result['source']}]"
        )

        print(
            result["content"]
        )

        print(
            f"\nSimilarity distance: "
            f"{result['distance']}"
        )

        print(
            "-" * 60
        )