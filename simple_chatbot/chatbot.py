"""
Chatbot
"""

import logging
from typing import List
from chromadb import Collection
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
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
    __sessions = {}
    __rules: List[SystemMessage] = []
    __rag_template = HumanMessagePromptTemplate.from_template(template="[Data]\n{data}\n[Question]\n{query}")

    def __init__(self, model: BaseLLM, embeddings: Embeddings, db: Collection, top_k: int = 5, memory_size: int = 20):
        self.__model = model
        self.__retreiver = _CustomRetriever(embeddings=embeddings, db=db, top_k=top_k)

    def add_rule(self, rule: str):
        self.__rules.append(SystemMessage(content=rule))

    def chat(self, query: str, session_id: str = "test_session") -> str:
        """
        Memory 필요
        """
        # rule -> histories -> current chat 순서로 모델에 입력된다.
        messages = [*self.__rules, MessagesPlaceholder(variable_name="history"), self.__rag_template]
        prompt = ChatPromptTemplate.from_messages(messages=messages)

        chain: Runnable = self.__get_chain(prompt=prompt)

        with_message_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=self.__get_session_history,
            input_messages_key="query",
            history_messages_key="history",
        )

        def format_docs(documents: List[Document]):
            return "\n\n".join(map(lambda x: x.page_content, documents))

        result = with_message_history.invoke(
            input={"data": (self.__retreiver | format_docs).invoke(query), "query": query},
            config={"configurable": {"session_id": session_id}},
        )
        chat_histories = self.__get_session_history(session_id).messages
        logging.debug(f"{chat_histories}")

        return result

    def __get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.__sessions:
            self.__sessions[session_id] = ChatMessageHistory()
        result: ChatMessageHistory = self.__sessions[session_id]
        return result

    def query(self, query: str) -> str:
        """
        단발성 질문
        """
        prompt = ChatPromptTemplate.from_messages(self.__rules + [self.__rag_template])

        chain: Runnable = self.__get_chain(prompt=prompt)
        result = chain.invoke(query)

        query = prompt.format(data=self.__retreiver.invoke(query), query=query)
        logging.debug(f"prompt: {query}\nanswer: {result}")

        return result

    def __get_chain(self, prompt: ChatPromptTemplate) -> Runnable:
        chain: Runnable = prompt | self.__model | StrOutputParser()
        return chain
