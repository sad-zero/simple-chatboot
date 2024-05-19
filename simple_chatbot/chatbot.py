"""
Chatbot
"""

import logging
from typing import List
from chromadb import Collection
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

    def get(self) -> List[BaseMessage]:
        return self.__core_db + self.__chat_db

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

    def __init__(self, model: BaseLLM, embeddings: Embeddings, db: Collection, top_k: int = 5, memory_size: int = 20):
        self.__model = model
        self.__embeddings = embeddings
        self.__db = db
        self.__top_k = top_k
        self.__memory = _Memory(limit=memory_size)

    def add_rule(self, rule: str):
        self.__memory.append_core_message(SystemMessage(content=rule))

    def chat(self, prompt: str) -> str:
        """
        Memory 필요
        """
        related_query: List[str] = self.__retreive_top_k(prompt)
        rag_prompt = self.__build_rag_prompt(prompt, related_query)

        histories = self.__memory.get()
        messages = histories + [rag_prompt]
        template = ChatPromptTemplate.from_messages(messages=messages)
        chain = template | self.__model | StrOutputParser()
        result = chain.invoke({"data": related_query, "prompt": prompt})
        self.__memory.append_chat_message(rag_prompt)
        self.__memory.append_chat_message(AIMessage(content=result))

        chat_historeis = histories + [rag_prompt.invoke({"data": related_query, "prompt": prompt}), result]

        logging.debug(f"{chat_historeis}")
        return result

    def query(self, prompt: str) -> str:
        """
        단발성 질문
        """
        related_query: List[str] = self.__retreive_top_k(prompt)
        rag_prompt = self.__build_rag_prompt()
        chain = rag_prompt | self.__model | StrOutputParser()

        query = rag_prompt.invoke({"data": related_query, "prompt": prompt})
        result = chain.invoke({"data": related_query, "prompt": prompt})
        logging.debug(f"prompt: {query}\nanswer: {result}")
        return result

    def __retreive_top_k(self, prompt: str) -> List[str]:
        embedded_prompt = self.__embeddings.embed_query(prompt)
        result = self.__db.query(query_embeddings=[embedded_prompt], n_results=self.__top_k)
        top_k_documents = [" ".join(d) for d in result["documents"]]
        return top_k_documents

    def __build_rag_prompt(self) -> BasePromptTemplate:
        prompt = ChatPromptTemplate.from_template(
            template="Using this data: {data}. Response to this prompt: {prompt}. If you don't find the response in the data, Response as \"I don't know\""
        )
        return prompt
