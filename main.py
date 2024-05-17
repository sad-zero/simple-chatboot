from langchain_community.embeddings.ollama import OllamaEmbeddings
import chromadb

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    documents = [
        "이 글은 라면에 대한 내용입니다.",
        "이 글은 짜장면에 대한 내용입니다.",
        "이 글은 햄버거에 대한 내용입니다.",
    ]
    client = chromadb.Client()
    collection = client.create_collection(name="docs")

    embedded = embeddings.embed_documents(documents)
