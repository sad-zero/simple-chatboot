from pprint import pprint
import pytest
import chromadb
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama

from simple_chatbot.chatbot import Chatbot


@pytest.fixture(scope="module")
def db():
    client = chromadb.PersistentClient("./resources/tests/vector_store.chroma")
    yield client.get_collection("books")


@pytest.fixture(scope="module")
def embeddings():
    embeddings = OllamaEmbeddings(model="all-minilm")
    yield embeddings


@pytest.fixture(scope="module")
def model():
    result = Ollama(model="llama3")
    yield result


def test_should_answer_when_query(model, db, embeddings):
    # given
    chatbot = Chatbot(embeddings=embeddings, db=db, model=model)
    prompt = "왜 집이 마음에 든 티를 내지 말아야 할까?"
    # when
    actual = chatbot.query(prompt=prompt)
    # then
    assert actual
