import certifi
import fitz
import re
import os
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")


def extract_text(pdf_path):  # PDF 텍스트 추출
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])


def extract_sections_from_text(text):  # 목차 기반 섹션 분할
    # 상위 섹션: 예) 1) General Principles
    top_section_pattern = r"(?m)^(?:\s*)?(\d+)\)\s+([A-Z][^\n]+)"
    top_sections = list(re.finditer(top_section_pattern, text))

    result = []

    for i, top in enumerate(top_sections):
        section_num = top.group(1).strip()
        section_title = top.group(2).strip()
        start = top.end()
        end = top_sections[i + 1].start() if i + 1 < len(top_sections) else len(text)

        section_text = text[start:end]

        # 하위 조항 분리: ex. 4.1, 4.2, ...
        clause_pattern = rf"(?m)^\s*({section_num}\.\d+)\s+[A-Z]"  # 4.1 + 대문자로 시작하는 경우만 패턴으로 인식
        clause_matches = list(re.finditer(clause_pattern, section_text))

        if clause_matches:
            for j, clause in enumerate(clause_matches):
                clause_num = clause.group(1).strip()
                clause_start = clause.end()
                clause_end = clause_matches[j + 1].start() if j + 1 < len(clause_matches) else len(section_text)

                content = section_text[clause_start - 1:clause_end].strip()  # -1인 패턴에 첫번쨰 대문자가 들어가 있어 -1로 조정

                result.append({
                    "section": section_num,
                    "sub_section": clause_num,
                    "title": section_title,
                    "content": content
                })
        else:
            # 조항 번호가 없을 경우 상위 섹션 전체 저장
            result.append({
                "section": section_num,
                "sub_section": None,
                "title": section_title,
                "content": section_text.strip()
            })

    return result


def create_documents(sections):  # LangChain Document 생성
    return [
        Document(
            page_content=sec["content"],
            metadata={
                "section": sec["section"],
                "sub_section": sec["sub_section"],
                "title": sec["title"]
            }
        )
        for sec in sections
    ]


def store_to_opensearch(documents):  # Opensearch에 저장
    embeddings = OpenAIEmbeddings()
    index_name = "f1_sporting_regulations"  # Opensearch 인덱스 이름

    # OpenSearch 클라이언트 구성
    opensearch_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_URL, "port": 443}],
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection
    )

    print("OpenSearch 연결 테스트 중...")
    print(opensearch_client.info())

    print("벡터 스토어 연결 중...")
    vector_store = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL + ":443",
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=60  # 업로드 안되는 이유: 기본 디폴트 timeout이 짧게 되어 있어서 -> 해결: timeout 늘림
    )

    print("문서 벡터 저장 중...")
    vector_store.add_documents(documents)

    # vector_store = OpenSearchVectorSearch.from_documents(
    #     documents=documents,
    #     embedding=embeddings,
    #     opensearch_client=opensearch_client,
    #     index_name=index_name,
    #     engine="faiss"
    # )

    return vector_store


if __name__ == "__main__":
    pdf_path = "F1_Rulebook_ver20250619/FIA 2025 Formula 1 Sporting Regulations - Issue 5 - 2025-04-30.pdf"
    print("PDF에서 텍스트 추출 중...")
    full_text = extract_text(pdf_path)

    print("목차 기반 섹션 분할 중...")
    sections = extract_sections_from_text(full_text)

    print(f"총 {len(sections)}개 섹션 분할 완료.")

    # (선택) 중간 결과 JSON 저장
    with open("sections.json", "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    print("문서 → LangChain Document 변환 중...")
    documents = create_documents(sections)

    print("Opensearch에 저장 중...")
    store_to_opensearch(documents)

    print("저장 완료!")
