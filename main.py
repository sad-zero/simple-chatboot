from pprint import pprint
from langchain_community.embeddings.ollama import OllamaEmbeddings
import chromadb

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="llama3")
    documents = [
        "이 글은 라면에 대한 내용입니다.",
        "이 글은 짜장면에 대한 내용입니다.",
        "이 글은 햄버거에 대한 내용입니다.",
    ]
    client = chromadb.PersistentClient("./resources/vector_store.chroma")
    collection = client.get_or_create_collection(name="docs")
    collection.update()

    # embedded_documents = embeddings.embed_documents(documents)

    # collection.add(
    #     ids=[str(i) for i in range(len(embedded_documents))], embeddings=embedded_documents, documents=documents
    # )

    prompt = "라면?"
    embedded_prompt = embeddings.embed_query(prompt)
    result = collection.query(query_embeddings=[embedded_prompt], n_results=1)
    pprint(result)
