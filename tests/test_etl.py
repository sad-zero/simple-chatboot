import json
import os
from typing import Dict, List

from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_community.embeddings.ollama import OllamaEmbeddings
import pytest
from simple_chatbot.etl import Extractor, Loader, Transformer
from simple_chatbot.vo import DocumentPair


@pytest.fixture(scope="module")
def src_path():
    yield "resources/references/book.pdf"


def test_should_extract_pdf_to_documents(src_path):
    # given
    extractor = Extractor()

    # when
    result = extractor.extract(pdf_path=src_path)

    # then
    assert len(result) > 0


@pytest.fixture(scope="function")
def pages(src_path):
    extractor = Extractor()
    yield extractor.extract(src_path)


@pytest.fixture(scope="function")
def embeddings():
    result = OllamaEmbeddings(model="all-minilm")
    yield result


def test_should_convert_page_to_chunk(pages):
    # given
    transformer = Transformer()
    # when
    actual = transformer.transform(pages)
    # then
    assert len(actual) > len(pages)


@pytest.fixture(scope="function")
def collection():
    client = PersistentClient(path="./resources/vector_store.chroma", settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection("books")


@pytest.fixture(scope="function")
def transformed_data(pages, embeddings):
    transformer = Transformer(embeddings=embeddings)
    result = transformer.transform(pages)
    return result


def test_should_store_embedding(collection, transformed_data):
    # given
    loader = Loader(collection=collection)
    expected = len(transformed_data)
    # when
    loader.load(transformed_data)
    actual = collection.count()
    # then
    assert actual == expected
