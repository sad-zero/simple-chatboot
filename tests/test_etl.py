import json
import os
from typing import Dict

from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_community.embeddings.ollama import OllamaEmbeddings
import pytest
from simple_chatbot.etl import Extractor, Loader, Transformer
from simple_chatbot.vo import DocumentPair


def test_should_extract_pdf_to_documents():
    # given
    src_path = "resources/references/book.pdf"
    extractor = Extractor()
    # when
    result = extractor.extract(pdf_path=src_path)

    # then
    assert len(result) > 0


@pytest.fixture(scope="function")
def normalized_data():
    with open("resources/tests/references/normalized_book.json", "r") as fd:
        data = json.load(fd)
        yield data


@pytest.fixture(scope="function")
def embeddings():
    result = OllamaEmbeddings(model="all-minilm")
    yield result


def test_should_convert_text_to_embedding(normalized_data, embeddings):
    # given
    transformer = Transformer(embeddings=embeddings)
    # when
    actual: Dict[int, DocumentPair] = transformer.transform(normalized_data)
    # then
    assert actual


@pytest.fixture(scope="function")
def collection():
    client = PersistentClient(path="./resources/vector_store.chroma", settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection("books")


@pytest.fixture(scope="function")
def transformed_data(normalized_data, embeddings):
    transformer = Transformer(embeddings=embeddings)
    result = transformer.transform(normalized_data)
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
