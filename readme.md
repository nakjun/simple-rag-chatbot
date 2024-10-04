# Simple RAG Chatbot

이 프로젝트는 PDF 문서를 기반으로 한 간단한 RAG (Retrieval-Augmented Generation) 챗봇 애플리케이션입니다.

## Demo
https://simple-rag-chatbot.streamlit.app/

## 주요 기능

- PDF 파일 업로드 및 처리
- 텍스트 임베딩 및 벡터 데이터베이스 생성
- 사용자 질문에 대한 관련 문서 검색
- OpenAI GPT 모델을 사용한 답변 생성
- 임베딩 비용 및 토큰 수 계산

## 사용된 기술

- Streamlit: 웹 인터페이스 구현
- LangChain: 텍스트 처리 및 OpenAI 모델 연동
- FAISS: 벡터 데이터베이스 및 유사도 검색
- OpenAI API: 텍스트 임베딩 및 챗봇 응답 생성

## 설치 방법

1. 필요한 라이브러리 설치:
   ```
   pip install -r requirements.txt
   ```

2. OpenAI API 키 설정:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## 실행 방법

다음 명령어로 애플리케이션을 실행합니다:
```
streamlit run app.py
```

## 사용 방법

1. 사이드바에서 임베딩 모델을 선택합니다 (large 또는 small).
2. PDF 파일을 업로드합니다.
3. 채팅 인터페이스에서 질문을 입력합니다.
4. 챗봇의 응답을 확인합니다.

## 주의사항

- OpenAI API 사용에 따른 비용이 발생할 수 있습니다.
- 대용량 PDF 파일 처리 시 시간이 걸릴 수 있습니다.

## 연락처
- njsung1217@gmail.com
- https://github.com/nakjun