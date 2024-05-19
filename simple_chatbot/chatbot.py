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
from langchain_core.runnables import Runnable, RunnablePassthrough


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


from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class _CustomRetriever(BaseRetriever):
    """
    BaseRetriever에서 subclass에 대한 속성 초기화를 담당한다. 따라서 초기화 메서드를 작성하면 안되고, public으로 properties를 선언해야 한다.
    """

    embeddings: Embeddings
    db: Collection
    top_k: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        embedded_prompt = self.embeddings.embed_query(query)
        result = self.db.query(query_embeddings=[embedded_prompt], n_results=self.top_k)
        top_k_documents: List[str] = [" ".join(d) for d in result["documents"]]
        result: List[Document] = []
        for content in top_k_documents:
            document = Document(page_content=content)
            result.append(document)
        return result


class Chatbot:
    __model: BaseLLM
    __retreiver: BaseRetriever
    __memory: _Memory
    __rag_template = HumanMessagePromptTemplate.from_template(template="[Data]\n{data}\n[Question]\n{query}")

    def __init__(self, model: BaseLLM, embeddings: Embeddings, db: Collection, top_k: int = 5, memory_size: int = 20):
        self.__model = model
        self.__retreiver = _CustomRetriever(embeddings=embeddings, db=db, top_k=top_k)
        self.__memory = _Memory(limit=memory_size)

    def add_rule(self, rule: str):
        self.__memory.append_core_message(SystemMessage(content=rule))

    def chat(self, query: str) -> str:
        """
        Memory 필요
        """
        core_messages = self.__memory.get_core()
        chat_messages = self.__memory.get_chat()
        messages = core_messages + chat_messages + [self.__rag_template]
        prompt = ChatPromptTemplate.from_messages(messages=messages)

        chain: Runnable = self.__get_chain(prompt=prompt)
        result = chain.invoke(query)

        query = self.__rag_template.format(data=self.__retreiver.invoke(query), query=query)
        self.__memory.append_chat_message(query)
        self.__memory.append_chat_message(AIMessage(content=result))

        chat_histories = core_messages + chat_messages + [query, result]
        logging.debug(f"{chat_histories}")

        return result

    def query(self, query: str) -> str:
        """
        단발성 질문
        """
        prompt = ChatPromptTemplate.from_messages(self.__memory.get_core() + [self.__rag_template])

        chain: Runnable = self.__get_chain(prompt=prompt)
        result = chain.invoke(query)

        query = prompt.format(data=self.__retreiver.invoke(query), query=query)
        logging.debug(f"prompt: {query}\nanswer: {result}")

        return result

    def __get_chain(self, prompt: ChatPromptTemplate) -> Runnable:
        def format_docs(documents: List[Document]):
            return "\n\n".join(map(lambda x: x.page_content, documents))

        chain: Runnable = (
            {"data": self.__retreiver | format_docs, "query": RunnablePassthrough()}
            | prompt
            | self.__model
            | StrOutputParser()
        )
        return chain
