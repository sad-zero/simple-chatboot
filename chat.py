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
    embeddings = OllamaEmbeddings(model="phi3:3.8b-mini-instruct-4k-q4_K_M")
    client = PersistentClient(path="resources/chroma_db", settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("books")
    model = ChatOllama(model="phi3:3.8b-mini-instruct-4k-q4_K_M")

    chatbot = Chatbot(embeddings=embeddings, db=collection, model=model)
    chatbot.add_rule("You should find reasons of your answer in the given data")
    chatbot.add_rule("You should answer with reasons")
    chatbot.add_rule('You should answer only "모르겠습니다" if you can\'t find the reasons in the given data')
    chatbot.add_rule("You should answer in korean, should not answer in english")
    chatbot.add_rule(
        """
[Examples]
Q1)
[Data]
자취 생활에는 다이소를 자주 이용합니다.
[Question]
자취 꿀팁 알려줘
A1)
제공된 내용에 따르면 자취시에는 다이소를 자주 이용해야 합니다.
Q2)
[Data]
계약 전에 집을 꼼꼼히 봐야 합니다.
[Question]
집을 계약하기 전에 주의할 점이 뭐야?
A2)
제공된 내용에 따르면 집을 계약하기 전에는 집을 꼼꼼히 봐야 합니다.
Q3)
[Data]
자취시에는 상비약을 챙겨야 합니다.
[Question]
배고파
A3)
모르겠습니다
Q4)
[Data]
자취시에는 채소를 꼭 먹어야 합니다.
[Question]
내가 방금 어떤 질문을 했어?
A4)
바로 직전에 했던 질문은 "배고파"입니다.
""".strip()
    )
    while True:
        query = input("Query: ")
        answer = chatbot.chat(query)
        print(f"Answer: {answer}")
