"""
PDF -> Vector Store ETL Pipeline
"""

import asyncio
from dataclasses import asdict
import json
import os
import re
from typing import Dict, List

from chromadb import Collection
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader

from simple_chatbot.vo import DocumentPair


class Extractor:
    """
    PDF -> Normalized Text로 변환
    """

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

    def __normalize_book(
        self,
        src_path: str,
        dest_path: str,
    ) -> Dict[int, str]:
        stop_patterns = re.compile(r"(﻿|https://www.youtube.com/@jachinam)|\u200b")

        with open(src_path, "r") as fd:
            book = json.load(fd)

        result = {}
        for page, content in book.items():
            normalized_content = stop_patterns.sub("", content)
            result[page] = normalized_content
        if dest_path:
            os.makedirs(dest_path[: dest_path.rfind("/")], exist_ok=True)
            with open(dest_path, "w") as fd:
                json.dump(result, fd, ensure_ascii=False)
                print(f"Extractor: Dump to {dest_path}")
        return result


class Transformer:
    """
    Normalized Text -> Embedding으로 변환
    Ollama를 사용하는 경우, 다음 조건을 만족해야 한다.
    1. ollama가 local에서 serving 됨
    2. ollama model이 pull 되어 있음
    """

    __embeddings: Embeddings

    def __init__(self, embeddings: Embeddings):
        self.__embeddings = embeddings

    def transform(
        self, data: Dict[int, str], dest_path: str = "resources/references/transformed_data.json"
    ) -> Dict[int, DocumentPair]:
        """
        @param data: {page_index: page_content}
        @return [embedding, ...] (order by page_index)
        """
        text_data = [v for _, v in sorted(data.items(), key=lambda x: x[0])]
        embedded = asyncio.run(self.__aembed_documents(text_data))
        result = {}
        for idx, (document, embedding) in enumerate(zip(text_data, embedded)):
            result[idx] = DocumentPair(document=document, embedding=embedding)
        if dest_path:
            with open(dest_path, "w") as fd:
                json.dump({key: asdict(val) for key, val in result.items()}, fd, ensure_ascii=False)
                print(f"Transformer: Dump to {dest_path}")
        return result

    async def __aembed_documents(self, text_data):
        coros = []
        for doc in text_data:
            coros.append(self.__embeddings.aembed_query(doc))
        result = await asyncio.gather(*coros)
        return result


class Loader:
    """
    Embedding -> Chroma DB 저장
    """

    __collection: Collection

    def __init__(self, collection: Collection):
        self.__collection = collection

    def load(self, data: Dict[int, DocumentPair], force: bool = True):
        """
        @param force: 이전 내용을 지우고 다시 만든다.
        """
        if force:
            self.__collection.delete(where={"tag": "simple_chatbot"})
        for idx, document_pair in data.items():
            self.__collection.add(
                ids=[str(idx)], embeddings=[document_pair.embedding], documents=[document_pair.document]
            )
        return
