from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
import pytest
from simple_chatbot.etl import Extractor, Loader, Transformer


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


def test_should_convert_page_to_chunk(pages):
    # given
    transformer = Transformer()
    # when
    actual = transformer.transform(pages)
    # then
    assert len(actual) > len(pages)


@pytest.fixture(scope="function")
def embeddings():
    result = OllamaEmbeddings(model="all-minilm")
    yield result


@pytest.fixture(scope="module")
def embedding_dir():
    return "/Users/dev/Documents/simple-chatboot/resources/tests/chroma_db"


@pytest.fixture(scope="function")
def chunked_data(pages):
    transformer = Transformer()
    result = transformer.transform(pages)
    return result


@pytest.fixture(scope="function")
def vector_store(embedding_dir, embeddings):
    store = InMemoryByteStore()
    embeddings_func = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings, document_embedding_cache=store, namespace="chat"
    )
    client = PersistentClient(path=embedding_dir, settings=Settings(anonymized_telemetry=False))
    chroma = Chroma(
        collection_name="books",
        embedding_function=embeddings_func,
        client=client,
        create_collection_if_not_exists=True,
    )
    yield chroma


def test_should_store_embedding(chunked_data, vector_store):
    # given
    loader = Loader(vector_store=vector_store)
    expected = len(chunked_data)
    # when
    loader.load(chunked_data, force=True)
    # then
    assert vector_store._collection.count() == len(chunked_data)
