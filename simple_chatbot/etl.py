"""
PDF -> Vector Store ETL Pipeline
"""

import asyncio
from dataclasses import asdict
import json
import logging
import os
import re
from typing import Dict, List

from chromadb import Collection, GetResult
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from chromadb.config import Settings

from simple_chatbot.vo import DocumentPair


class Extractor:

    def extract(
        self,
        pdf_path: str = "./resources/references/book.pdf",
    ) -> List[Document]:
        """
        @return {page_index: page_content}
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        return pages


class Transformer:
    """
    List[Document] -> Chunks
    """

    __text_splitter: RecursiveCharacterTextSplitter

    def __init__(self):
        self.__text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

    def transform(self, data: List[Document]) -> List[Document]:
        """
        @param data: Paged Documents
        @return Chunked Documents
        """
        str_documents: List[str] = list(map(lambda x: x.page_content, data))
        metadatas = list(map(lambda x: x.metadata, data))
        result = self.__text_splitter.create_documents(str_documents, metadatas=metadatas)
        return result


class Loader:
    """
    Embedding -> Chroma DB 저장
    """

    __vector_store: Chroma

    def __init__(self, vector_store: Chroma):
        self.__vector_store = vector_store

    def load(self, data: List[Document], force: bool = True):
        """
        @param force: 이전 내용을 지우고 다시 만든다.
        """
        if force:
            collection_name = self.__vector_store._collection.name
            self.__vector_store.delete_collection()
            self.__vector_store._collection = self.__vector_store._client.create_collection(collection_name)
        self.__vector_store.add_documents(documents=data)
        return
