import os
import uuid
import pandas as pd
import chromadb


class Portfolio:
    def __init__(self, file_path="resourse/my_portfolio.csv"):
        # Resolve absolute path safely (cloud-safe)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(current_dir, file_path)

        self.data = pd.read_csv(self.file_path)

        # Cloud-safe persistent path
        self.chroma_client = chromadb.PersistentClient(path="./vectorstore")
        self.collection = self.chroma_client.get_or_create_collection(
            name="portfolio"
        )

    def load_portfolio(self):
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=row["Techstack"],
                    metadatas={"links": row["Links"]},
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        result = self.collection.query(
            query_texts=skills,
            n_results=2
        )
        return result.get("metadatas", [])
