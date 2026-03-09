from google.genai.types import Retrieval
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client.http import models as qmodels

from project import config
from qdrant_client import QdrantClient

class VectorDbManager:
    __client: QdrantClient
    __dense_embedding: HuggingFaceEmbeddings
    __sparse_embedding: FastEmbedSparse
    def __init__(self):
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__dense_embedding = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        self.__sparse_embedding = FastEmbedSparse(model_name=config.SPARSE_MODEL)

    def create_collection(self, collection_name):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=len(self.__dense_embedding.embed_query("test")), distance=qmodels.Distance.COSINE),
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
            )
            print(f"Collection {collection_name} created.")
        else:
            print(f"Collection {collection_name} already exists.")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing collection {collection_name}...")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,
                embedding=self.__dense_embedding,
                sparse_embedding=self.__sparse_embedding,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_vector_name=config.SPARSE_VECTOR_NAME,
            )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")