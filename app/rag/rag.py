import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# Connect to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Connect to our embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
db = chromadb.PersistentClient(path="chroma_db")

collection = db.get_collection(
    name="project_documents"
)


def retrieve(query):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    return results["documents"][0]


def generate_answer(question, context):
    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = input("Ask a question: ")

    documents = retrieve(question)

    context = "\n\n".join(documents)

    answer = generate_answer(question, context)

    print("\nAnswer:\n")
    print(answer)