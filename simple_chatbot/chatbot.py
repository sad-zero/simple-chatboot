"""
Chatbot
"""

from copy import copy
import logging
from typing import List
from chromadb import Collection
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable


class _Memory:
    __core_db: List[SystemMessage] = []
    __chat_db: List[BaseMessage] = []
    __limit: int = 20

    def __init__(self, limit: int = 20):
        if limit:
            self.__limit = limit

    def append_core_message(self, message: SystemMessage):
        self.__core_db.append(message)

    def append_chat_message(self, message: AIMessage | HumanMessage):
        """
        Store latest n-messages
        """
        if len(self.__chat_db) > self.__limit:
            self.__chat_db.pop(0)
        self.__chat_db.append(message)

    def get_core(self) -> List[SystemMessage]:
        return copy(self.__core_db)

    def get_chat(self) -> List[HumanMessage | AIMessage]:
        return copy(self.__chat_db)

    def clear_chat(self):
        self.__chat_db.clear()

    def clear_core(self):
        self.__core_db.clear()


class Chatbot:
    __model: BaseLLM
    __embeddings: Embeddings
    __db: Collection
    __top_k: int
    __memory: _Memory
    __rag_template = HumanMessagePromptTemplate.from_template(
        template="Using this data: {data}. Response to this question: {query}"
    )

    def __init__(self, model: BaseLLM, embeddings: Embeddings, db: Collection, top_k: int = 5, memory_size: int = 20):
        self.__model = model
        self.__embeddings = embeddings
        self.__db = db
        self.__top_k = top_k
        self.__memory = _Memory(limit=memory_size)

    def add_rule(self, rule: str):
        self.__memory.append_core_message(SystemMessage(content=rule))

    def chat(self, query: str) -> str:
        """
        Memory 필요
        """
        related_query: List[str] = self.__retreive_top_k(query)
        core_messages = self.__memory.get_core()
        chat_messages = self.__memory.get_chat()
        messages = core_messages + chat_messages + [self.__rag_template]
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        chain: Runnable = prompt | self.__model | StrOutputParser()

        result = chain.invoke({"data": related_query, "query": query})

        query = self.__rag_template.format(data=related_query, query=query)
        self.__memory.append_chat_message(query)
        self.__memory.append_chat_message(AIMessage(content=result))

        chat_histories = core_messages + chat_messages + [query, result]
        logging.debug(f"{chat_histories}")

        return result

    def query(self, query: str) -> str:
        """
        단발성 질문
        """
        related_query: List[str] = self.__retreive_top_k(query)
        prompt = ChatPromptTemplate.from_messages(self.__memory.get_core() + [self.__rag_template])

        chain: Runnable = prompt | self.__model | StrOutputParser()
        result = chain.invoke({"data": related_query, "query": query})

        query = prompt.format(data=related_query, query=query)
        logging.debug(f"prompt: {query}\nanswer: {result}")

        return result

    def __retreive_top_k(self, prompt: str) -> List[str]:
        embedded_prompt = self.__embeddings.embed_query(prompt)
        result = self.__db.query(query_embeddings=[embedded_prompt], n_results=self.__top_k)
        top_k_documents = [" ".join(d) for d in result["documents"]]
        return top_k_documents
