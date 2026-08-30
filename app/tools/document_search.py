from app.rag.retrieve import search


def search_documents(query: str):
    """
    Search the knowledge base and return
    context together with source information.
    """

    results = search(
        query,
        n_results=3
    )

    if not results:

        return {
            "context": (
                "No relevant information was found "
                "in the knowledge base."
            ),
            "sources": []
        }


    context_parts = []
    sources = []


    for result in results:

        context_parts.append(
            (
                f"Source: {result['source']}\n"
                f"Category: {result['category']}\n\n"
                f"Content:\n{result['content']}"
            )
        )


        source_info = {
            "source": result["source"],
            "category": result["category"]
        }


        if source_info not in sources:

            sources.append(
                source_info
            )


    return {

        "context": "\n\n".join(
            context_parts
        ),

        "sources": sources
    }