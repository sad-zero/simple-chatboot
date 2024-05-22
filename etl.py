"""
ETL Pipeline
"""

import logging
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_chroma import Chroma
from simple_chatbot.etl import Extractor, Loader, Transformer
from langchain.storage import InMemoryByteStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_vector_store(coll_name: str = "books", persistent_path: str = "resources/chroma_db") -> Chroma:
    embeddings = OllamaEmbeddings(model="phi3:3.8b-mini-instruct-4k-q4_K_M")
    store = InMemoryByteStore()
    embeddings_func = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings, document_embedding_cache=store, namespace="in-memory-chat"
    )
    client = PersistentClient(path=persistent_path, settings=Settings(anonymized_telemetry=False))
    result = Chroma(
        collection_name=coll_name,
        embedding_function=embeddings_func,
        client=client,
        create_collection_if_not_exists=True,
    )
    return result


logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    extractor = Extractor()
    transformer = Transformer()
    loader = Loader(vector_store=get_vector_store())

    logging.info("Phase1: extract from pdf")
    data = extractor.extract()
    logging.info("Phase2: transform documents")
    data = transformer.transform(data)
    logging.info("Phase3: load to vector store")
    loader.load(data)
    logging.info("Done")
