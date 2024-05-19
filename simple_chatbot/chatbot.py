"""
Chatbot
"""

import logging
from typing import List
from chromadb import Collection
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.llms import BaseLLM


class Chatbot:
    __model: BaseLLM
    __embeddings: Embeddings
    __db: Collection
    __top_k: int

    def __init__(self, model: BaseLLM, embeddings: Embeddings, db: Collection, top_k: int = 5):
        self.__model = model
        self.__embeddings = embeddings
        self.__db = db
        self.__top_k = top_k

    def chat(self, prompt: str):
        """
        Memory 필요
        """
        pass

    def query(self, prompt: str) -> str:
        """
        단발성 질문
        """
        related_query: List[str] = self.__retreive_top_k(prompt)
        rag_prompt = self.__build_rag_prompt(prompt, related_query)
        result = self.__model.invoke(rag_prompt)

        logging.debug(f"prompt: {rag_prompt}\nanswer: {result}")
        return result

    def __retreive_top_k(self, prompt: str) -> List[str]:
        embedded_prompt = self.__embeddings.embed_query(prompt)
        result = self.__db.query(query_embeddings=[embedded_prompt], n_results=self.__top_k)
        top_k_documents = [" ".join(d) for d in result["documents"]]
        return top_k_documents

    def __build_rag_prompt(self, prompt: str, data: List[str]) -> str:
        prompt = f"Using this data: {data}. Response to this prompt: {prompt}. If you don't find the response in the data, Response as \"I don't know\""
        return prompt
