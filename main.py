from chromadb import PersistentClient
from simple_chatbot.chatbot import Chatbot
from simple_chatbot.etl import Extractor, Loader, Transformer
from langchain_community.embeddings.ollama import OllamaEmbeddings


if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="all-minilm")
    client = PersistentClient(path="./resources/vector_store.chroma")
    collection = client.get_or_create_collection("books")

    # extractor = Extractor()
    # transformer = Transformer(embeddings=embeddings)
    # loader = Loader(collection=collection)

    # print("===Extractor===")
    # data = extractor.extract()
    # print("===Transform==")
    # data = transformer.transform(data)
    # print("===Loader==")
    # loader.load(data)
