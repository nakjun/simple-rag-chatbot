import os
import warnings
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pdf_to_md import convert_pdf_to_markdown
import torch
import tiktoken
import openai

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine 'expire_cache' was never awaited")

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 임베딩 모델 초기화 함수
@st.cache_resource
def initialize_embedding_model(model_name):
    return OpenAIEmbeddings(model=model_name)

# 토큰 수 계산 함수
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# PDF 처리 및 벡터 DB 생성 함수
def process_pdf(pdf_file, embedding_model):
    # PDF를 Markdown으로 변환
    markdown_content = convert_pdf_to_markdown(pdf_file)
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(markdown_content)
    
    # 토큰 수 계산
    total_tokens = sum(num_tokens_from_string(text, "cl100k_base") for text in texts)
    
    # 벡터 데이터베이스 생성
    vectorstore = FAISS.from_texts(texts, embedding_model)
    return vectorstore, total_tokens

# 비용 계산 함수
def calculate_embedding_cost(total_tokens, model_name):
    if model_name == "text-embedding-3-small":
        return total_tokens * 0.00002 / 1000  # $0.02 per 1000 tokens
    elif model_name == "text-embedding-3-large":
        return total_tokens * 0.00013 / 1000  # $0.13 per 1000 tokens
    else:
        return 0

# OpenAI 모델 초기화
@st.cache_resource
def init_openai_model():
    return ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.001)

# 관련 문서 검색
def retrieve_docs(vectordb, query: str, k: int = 5):
    docs_and_scores = vectordb.similarity_search_with_score(query, k=k)
    
    results = []
    for doc, score in docs_and_scores:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": score
        })
    
    return results

# Streamlit 앱 메인 함수
def main():
    st.title("Simple RAG Chatbot")

    with st.sidebar:
        st.title("RAG Chatbot")
        st.markdown("---")
        st.markdown("Developed by [nakjun](https://github.com/nakjun)")
        embedding_option = st.selectbox("임베딩 모델 선택", ("large", "small"))
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

    if embedding_option == "large":
        embedding_model = initialize_embedding_model("text-embedding-3-large")
        model_name = "text-embedding-3-large"
    else:
        embedding_model = initialize_embedding_model("text-embedding-3-small")
        model_name = "text-embedding-3-small"

    if uploaded_file is not None:
        with st.spinner("PDF 파일을 처리 중입니다..."):
            vectordb, total_tokens = process_pdf(uploaded_file, embedding_model)
        st.success("업로드한 PDF 파일을 성공적으로 처리하였습니다.")

        # 임베딩 비용 계산
        embedding_cost = calculate_embedding_cost(total_tokens, model_name)

        # 사이드바에 토큰 수와 비용 표시
        with st.sidebar:
            st.markdown("---")
            st.subheader("임베딩 정보")
            st.write(f"총 토큰 수: {total_tokens:,} by tiktoken")
            st.write(f"예상 비용: ${embedding_cost:.4f} by openai")

        # 채팅 기록을 저장할 세션 상태 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=("🧑‍💻" if message["role"] == "user" else "🤖")):
                st.markdown(message["content"])

        # 사용자 입력 받기
        st.markdown("---")
        if query_text := st.chat_input("메시지를 입력하세요."):
            st.session_state.messages.append({"role": "user", "content": query_text})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(query_text)

            with st.spinner("RAG 엔진을 이용해 관련 문서를 검색 중입니다..."):
                results = retrieve_docs(vectordb, query_text)

                # 사이드바에 결과 출력
                with st.sidebar:
                    st.subheader("검색된 문서:")
                    for i, result in enumerate(results, 1):
                        st.markdown(f"**문서 {i}**")
                        st.write(f"내용: {result['content'][:400]}...")
                        st.write(f"유사도: {result['similarity_score']}")
                        st.markdown("---")

            if results:
                context = "\n\n".join([f"{result['content']}" for result in results])
                prompt = f"""다음은 주어진 질문에 대한 참고 자료입니다:

                {context}

                전달한 정보를 바탕으로 다음 질문에 대해 상세하고 정확하게 답변해주세요:

                질문: {query_text}

                답변 시 다음 지침을 따라주세요:
                1. 제공된 정보만을 사용하여 답변하세요.
                2. 정보가 불충분하거나 관련이 없는 경우, 그 사실을 명시하세요.
                3. 추측하지 말고, 확실한 정보만 제공하세요.
                4. 답변은 논리적이고 구조화된 형식으로 작성하세요.
                5. 전문 용어가 있다면 간단히 설명을 덧붙이세요.
                6. 답변은 한글로 작성하세요.

                답변:"""

                with st.spinner("답변을 생성 중입니다..."):
                    try:
                        openai_model = init_openai_model()
                        response = openai_model.predict(prompt)
                        
                        # AI 응답 추가
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(response)

                    except Exception as e:
                        st.error(f"텍스트 생성 중 오류가 발생했습니다: {str(e)}")
            else:
                st.warning("관련 문서를 찾을 수 없습니다.")
    else:
        st.info("PDF 파일을 업로드해주세요.")

if __name__ == "__main__":
    main()