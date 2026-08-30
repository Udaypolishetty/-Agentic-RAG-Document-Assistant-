import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="project_documents"
)


def search(query):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    return results["documents"][0]


if __name__ == "__main__":
    question = input("Ask a question: ")

    results = search(question)

    print("\nRelevant information:\n")

    for result in results:
        print(result)