from copy import deepcopy
import json
import os
from typing import Dict

import chromadb
from langchain_community.embeddings.ollama import OllamaEmbeddings
import numpy as np
import pytest
from simple_chatbot.etl import Extractor, Loader, Transformer
from simple_chatbot.vo import DocumentPair


def test_should_extract_pdf_to_normalized_json():
    # given
    src_path = "resources/references/book.pdf"
    raw_dest_path = "resources/tests/references/book.json"
    dest_path = "resources/tests/references/normalized_book.json"
    extractor = Extractor()
    # when
    result = extractor.parse(pdf_path=src_path, raw_dest_path=raw_dest_path, dest_path=dest_path)

    # then
    assert os.path.exists(raw_dest_path)
    assert os.path.exists(dest_path)

    with open(dest_path, "r") as fd:
        actual = json.load(fd)
    assert actual == result


@pytest.fixture(scope="function")
def normalized_data():
    with open("resources/tests/references/normalized_book.json", "r") as fd:
        data = json.load(fd)
        yield data


@pytest.fixture(scope="function")
def embeddings():
    result = OllamaEmbeddings(model="llama3")
    yield result


def test_should_convert_text_to_embedding(normalized_data, embeddings):
    # given
    transformer = Transformer(embeddings=embeddings)
    expected = {}
    sorted_documents = [v for _, v in sorted(normalized_data.items(), key=lambda x: x[0])]
    embeddeds = embeddings.embed_documents(sorted_documents)
    for idx, (document, embedding) in enumerate(zip(sorted_documents, embeddeds)):
        expected[idx] = DocumentPair(document=document, embedding=embedding)
    # when
    actual: Dict[int, dict] = transformer.transform(normalized_data)
    # then
    for k in actual.keys():
        assert np.array_equal(actual[k].embedding, expected[k].embedding)


@pytest.fixture(scope="function")
def collection():
    client = chromadb.PersistentClient(path="./resources/tests/vector_store.chroma")
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
