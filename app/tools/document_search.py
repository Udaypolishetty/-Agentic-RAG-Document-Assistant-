from app.rag.rag import retrieve


def search_documents(query):
    """Search the project knowledge base."""
    return retrieve(query)