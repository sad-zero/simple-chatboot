"""
PDF -> Vector Store ETL Pipeline
"""

import json
import os
import re
from typing import Dict

from pypdf import PdfReader


class Extractor:
    def parse(
        self,
        pdf_path: str = "./resources/references/book.pdf",
        raw_dest_path: str = "./resources/references/book.json",
        dest_path: str = "./resources/references/normalized_book.json",
    ):
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
