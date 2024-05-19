from pprint import pprint
import pytest
import chromadb
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.chat_models.ollama import ChatOllama

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
    model = ChatOllama(model="phi3")
    yield model


@pytest.fixture(scope="module")
def chatbot(model, db, embeddings):
    chatbot = Chatbot(embeddings=embeddings, db=db, model=model)
    chatbot.add_rule("You should find reasons of your answer in the given data")
    chatbot.add_rule("You should answer with reasons")
    chatbot.add_rule('You should answer only "모르겠습니다" if you can\'t find the reasons in the given data')
    chatbot.add_rule("You should answer in korean")
    return chatbot


def test_should_answer_when_query(chatbot):
    # given
    prompt = "왜 집이 마음에 든 티를 내지 말아야 할까?"
    # when
    actual = chatbot.query(query=prompt)
    # then
    assert actual
