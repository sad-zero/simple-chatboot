"""
Chatbot
"""

import logging

from chromadb import PersistentClient
from chromadb.config import Settings
from simple_chatbot.chatbot import Chatbot
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="all-minilm")
    client = PersistentClient(path="./resources/vector_store.chroma", settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("books")
    model = ChatOllama(model="phi3")

    chatbot = Chatbot(embeddings=embeddings, db=collection, model=model)
    chatbot.add_rule("You should find reasons of your answer in the given data")
    chatbot.add_rule("You should answer with reasons")
    chatbot.add_rule('You should answer only "모르겠습니다" if you can\'t find the reasons in the given data')
    chatbot.add_rule("You should answer in korean")
    while True:
        query = input("Query: ")
        answer = chatbot.chat(query)
        print(f"Answer: {answer}")
