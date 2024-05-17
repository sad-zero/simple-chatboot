import json
import os

from langchain_community.embeddings.ollama import OllamaEmbeddings
import numpy as np
import pytest
from simple_chatbot.etl import Extractor, Transformer


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
    result = OllamaEmbeddings(model="mxbai-embed-large")
    yield result


def test_should_convert_text_to_ndarray(normalized_data, embeddings):
    # given
    transformer = Transformer(embeddings=embeddings)
    expected = np.array(embeddings.embed_documents([v for _, v in sorted(normalized_data.items(), key=lambda x: x[0])]))
    # when
    actual = transformer.transform(normalized_data)
    # then
    breakpoint()
    assert np.array_equal(actual, expected)
