"""
PDF -> Vector Store ETL Pipeline
"""

import json
import os
import re
from typing import Dict

from chromadb import Client, Collection
from langchain_core.embeddings.embeddings import Embeddings
from pypdf import PdfReader
import numpy as np

from simple_chatbot.vo import DocumentPair


class Extractor:
    """
    PDF -> Normalized Text로 변환
    """

    def extract(
        self,
        pdf_path: str = "./resources/references/book.pdf",
        raw_dest_path: str = "./resources/references/book.json",
        dest_path: str = "./resources/references/normalized_book.json",
    ) -> Dict[int, str]:
        """
        @return {page_index: page_content}
        """
        if not pdf_path or not raw_dest_path or not dest_path:
            raise RuntimeError(f"pdf_path: {pdf_path}, raw_dest_path: {raw_dest_path}, dest_path: {dest_path}")
        self.__parse_pdf_to_json(pdf_path=pdf_path, dest_path=raw_dest_path)
        result = self.__normalize_book(src_path=raw_dest_path, dest_path=dest_path)
        return result

    def __parse_pdf_to_json(self, pdf_path: str, dest_path: str) -> Dict[int, str]:
        pdf_reader = PdfReader(pdf_path)
        text_book = {}
        for idx, page in enumerate(pdf_reader.pages):
            text_page = page.extract_text()
            text_book[idx] = text_page

        if dest_path:
            os.makedirs(dest_path[: dest_path.rfind("/")], exist_ok=True)
            with open(dest_path, "w") as fd:
                json.dump(text_book, fd, ensure_ascii=False)

        return text_book

    def __normalize_book(
        self,
        src_path: str,
        dest_path: str,
    ) -> Dict[int, str]:
        stop_patterns = re.compile(r"(﻿|https://www.youtube.com/@jachinam)")

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

    def transform(self, data: Dict[int, str]) -> Dict[int, DocumentPair]:
        """
        @param data: {page_index: page_content}
        @return [embedding, ...] (order by page_index)
        """
        text_data = [v for _, v in sorted(data.items(), key=lambda x: x[0])]
        embedded = self.__embeddings.embed_documents(texts=text_data)
        result = {}
        for idx, (document, embedding) in enumerate(zip(text_data, embedded)):
            result[idx] = DocumentPair(document=document, embedding=embedding)
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
