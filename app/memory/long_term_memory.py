import chromadb


class LongTermMemory:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="data/memory"
        )

        self.collection = self.client.get_or_create_collection(
            name="long_term_memory"
        )

    def save(self, content):

        memory_id = str(
            self.collection.count() + 1
        )

        self.collection.add(
            documents=[content],
            ids=[memory_id]
        )

    def search(self, query, limit=3):

        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )

        return results["documents"][0]