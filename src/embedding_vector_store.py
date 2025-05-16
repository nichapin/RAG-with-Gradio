from dataclasses import dataclass
import os

import torch
import torchvision
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

@dataclass
class DataEmbedding:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"

    def embedding_vector_store(self,clean_competencies_df,role_df):
        # Competency documents
        comp_docs = [
                        Document(
                            page_content=row.description,
                            metadata={"type": "competency", "name": row.competency},
                        )
                        for _, row in clean_competencies_df.iterrows()
                    ]

        # Role documents (optional—same pattern)
        role_docs = [
                        Document(
                            page_content=row.description,
                            metadata={"type": "role", "name": row.role},
                        )
                        for _, row in role_df.iterrows()
                    ]

        # Combine if you want a single index
        all_docs = comp_docs + role_docs

        # Use a local HF embedding model (no API key needed)
        hf_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        # Build the FAISS index from your Documents
        db = FAISS.from_documents(all_docs, hf_embeddings)

        db.save_local(os.path.join("artifact","faiss_index"))

        return db
