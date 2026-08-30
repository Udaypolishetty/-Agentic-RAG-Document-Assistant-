import chromadb
from sentence_transformers import SentenceTransformer


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
# CONFIGURATION
# ==================================================

DEFAULT_RESULTS = 2

# Lower distance = better match
# SIMILARITY_THRESHOLD = 1.15


# ==================================================
# SEARCH
# ==================================================

def search(
    query: str,
    n_results: int = DEFAULT_RESULTS
):

    if not query or not query.strip():
        return []

    # Create query embedding
    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # Vector search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted_results = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        formatted_results.append({

            "content": document,

            "source": metadata.get(
                "source",
                "Unknown"
            ),

            "category": metadata.get(
                "category",
                "Knowledge Base"
            ),

            "distance": distance
        })

    return formatted_results


# ==================================================
# TEST SEARCH FROM TERMINAL
# ==================================================

if __name__ == "__main__":

    question = input(
        "Ask a question: "
    )


    results = search(
        question
    )


    print(
        "\nRelevant information:\n"
    )


    if not results:

        print(
            "No sufficiently relevant information found."
        )

    else:

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