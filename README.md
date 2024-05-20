# simple-chatboot
[자취생활백서](https://m.cafe.naver.com/ca-fe/web/cafes/jachinam/articles/15233?useCafeId=false&amp;tc) 챗봇 만들기

## 설치
- python 3.10 가상환경 생성(Ex. conda)
- `pip install -e .`
- `ollama 설치`
- `ollama pull all-minilm` : Vector Store Embedding Model
- `ollama pull phi3`: Chat Model

## Vector Store 생성
1. `resources/references/book.pdf`에 위의 PDF 놓기
2. `python etl.py`
3. `resources/chroma_db` 폴더 생성 확인

## Chat
1. `python chat.py`
2. `CTRL + C` 로 종료
