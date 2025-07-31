import certifi
import fitz
import re
import os
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

patterns = {
    "default": {
        "top_section": r"(?m)^(?:\s*)?(\d+)\)\s+([A-Z][^\n]+)",
        "clause": r"(?m)^\s*({section_num}\.\d+)\s+[A-Z]"},
    "f1_sporting_regulations": {
        "top_section": r"(?m)^(?:\s*)?(\d+)\)\s+([A-Z][^\n]+)",
        "clause": r"(?m)^\s*({section_num}\.\d+)\s+[A-Z]"},
    "f1_pu_financial_regulations": {
        "top_section": r"(?m)^\s*(\d+)\.\s+([A-Z][^\n]+)",
        "clause": r"(?m)^\s*({section_num}\.\d+)\s+[A-Z]"},
    "f1_financial_regulations": {
        "top_section": r"(?m)^\s*(\d+)\.\s+([A-Z][^\n]+)",
        "clause": r"(?m)^\s*({section_num}\.\d+)\s+[A-Z]"},
    "f1_technical_regulations": {
        "top_section": r"(?m)^ARTICLE\s+(\d+):\s+([A-Z][^\n]+)",
        "clause": r"(?m)^\s*({section_num}\.\d+)(?!\.\d)\s+[A-Z]",
        "sub_clause": r"(?m)^\s*({sub_section_num}\.\d+)\s+[A-Z]"},
}
rename = {
    "Formula 1 Sporting Regulations": "f1_sporting_regulations",
    "Formula 1 PU Financial Regulations": "f1_pu_financial_regulations",
    "Formula 1 Financial Regulations": "f1_financial_regulations",
    "formula_1_technical_regulations": "f1_technical_regulations",
}

# OpenSearch 클라이언트 구성
opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_URL, "port": 443}],
    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection
)


def extract_text(pdf_path):  # PDF 텍스트 추출
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])


def extract_sections_from_text(text, top_section_pattern, clause_pattern_template):  # 목차 기반 섹션 분할
    # 상위 섹션: 예) 1. General Principles
    # top_section_pattern = r"(?m)^\s*(\d+)\.\s+([A-Z][^\n]+)"
    top_sections = list(re.finditer(top_section_pattern, text))

    result = []

    for i, top in enumerate(top_sections):
        section_num = top.group(1).strip()
        section_title = top.group(2).strip()
        start = top.end()
        end = top_sections[i + 1].start() if i + 1 < len(top_sections) else len(text)

        section_text = text[start:end]

        # 하위 조항 분리: ex. 4.1, 4.2, ...
        # clause_pattern = rf"(?m)^\s*({section_num}\.\d+)\s+[A-Z]"  # 4.1 + 대문자로 시작하는 경우만 패턴으로 인식
        clause_pattern = clause_pattern_template.format(section_num=section_num)
        clause_matches = list(re.finditer(clause_pattern, section_text))

        if clause_matches:
            for j, clause in enumerate(clause_matches):
                clause_num = clause.group(1).strip()
                clause_start = clause.end()
                clause_end = clause_matches[j + 1].start() if j + 1 < len(clause_matches) else len(section_text)

                content = section_text[clause_start - 1:clause_end].strip()  # -1인 패턴에 첫번째 대문자가 들어가 있어 -1로 조정

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


def extract_sections_from_text2(text, top_section_pattern_template, level1_pattern_template,
                                level2_pattern_template):  # 목차 기반 섹션 분할
    # 상위 섹션: 예) ARTICLE 1: GENERAL PRINCIPLES
    top_sections = list(re.finditer(top_section_pattern_template, text))

    result = []

    for i, top in enumerate(top_sections):
        section_num = top.group(1).strip()
        section_title = top.group(2).strip()
        start = top.end()
        end = top_sections[i + 1].start() if i + 1 < len(top_sections) else len(text)

        section_text = text[start-1:end].strip()

        # 1단계 조항(1.1) 매칭
        level1_pattern = level1_pattern_template.format(section_num=section_num)
        level1_matches = list(re.finditer(level1_pattern, section_text))

        if level1_matches:
            for i1, m1 in enumerate(level1_matches):
                sub_section_num = m1.group(1).strip()
                start1 = m1.end()
                end1 = level1_matches[i1 + 1].start() if i1 + 1 < len(level1_matches) else len(section_text)
                level1_text = section_text[start1-1:end1].strip()

                # 2단계 조항(1.1.1) 매칭
                level2_pattern = level2_pattern_template.format(sub_section_num=sub_section_num)
                level2_matches = list(re.finditer(level2_pattern, level1_text))
                if level2_matches:
                    for i2, m2 in enumerate(level2_matches):
                        sub_sub_section_num = m2.group(1).strip()
                        start2 = m2.end()
                        end2 = level2_matches[i2 + 1].start() if i2 + 1 < len(level2_matches) else len(level1_text)

                        content = level1_text[start2-1:end2]

                        result.append({
                            "section": section_num,
                            "sub_section": sub_section_num,
                            "sub_sub_section": sub_sub_section_num,
                            "title": section_title,
                            "content": content
                        })
                else:
                    # 2단계가 없으면 1단계 전체를 하나의 항목으로
                    content = level1_text.strip()
                    result.append({
                        "section": section_num,
                        "sub_section": sub_section_num,
                        "sub_sub_section": None,
                        "title": section_title,
                        "content": content
                    })
        else:
            # 조항 번호가 없을 경우 상위 섹션 전체 저장
            result.append({
                "section": section_num,
                "sub_section": None,
                "sub_sub_section": None,
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


def store_to_opensearch(documents, index_name):  # Opensearch에 저장
    embeddings = OpenAIEmbeddings()
    # index_name = "f1_sporting_regulations"  # Opensearch 인덱스 이름

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
    # vector_store.add_documents(documents)

    # Langchain documents를 OpenSearch에 저장하기 위한 actions 생성
    actions = []
    for doc in documents:
        actions.append({
            "_op_type": "index",
            "_index": vector_store.index_name,
            "_source": {
                **doc.metadata,
                "text": doc.page_content
            }
        })

    # parallel bulk 호출
    success, failed = 0, 0
    for ok, item in helpers.parallel_bulk(
            client=vector_store.client,  # OpenSearch 클라이언트 인스턴스
            actions=actions,
            thread_count=4,  # 워커 스레드 수 (예: 4)
            chunk_size=int(os.getenv("CHUNK_SIZE", 300)),  # 한 번에 보낼 도큐먼트 수 (기본값: 300, 환경 변수로 설정 가능)
            request_timeout=60  # 타임아웃 여유 있게
    ):
        if ok:
            success += 1
        else:
            failed += 1
            logging.error(f"Failed to index document: {item}")

    print(f"[✅] Indexed: {success}, Failed: {failed}")


    # vector_store = OpenSearchVectorSearch.from_documents(
    #     documents=documents,
    #     embedding=embeddings,
    #     opensearch_client=opensearch_client,
    #     index_name=index_name,
    #     engine="faiss"
    # )

    return vector_store


if __name__ == "__main__":
    pdf_dir = "F1_Rulebook_ver20250619/"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf') and not f.startswith('.')]
    # pdf_path = "F1_Rulebook_ver20250619/fia_2025_formula_1_technical_regulations_-_issue_03_-_2025-04-07.pdf"

    all_sections = []
    print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")

    for pdf_path in pdf_files:
        print(f"처리 중: {pdf_path}")

        # 파일 이름에서 패턴 키 선택
        selected_pattern_key = "default"
        sorted_keys = sorted(patterns.keys(), key=len, reverse=True)
        sorted_rename = sorted(rename.keys(), key=len, reverse=True)

        for rename_value in sorted_rename:
            if rename_value in pdf_path:
                selected_pattern_key = rename[rename_value]
                break

        print(f"선택된 패턴 키: {selected_pattern_key}")

        selected_patterns = patterns[selected_pattern_key]
        top_section_pattern = selected_patterns["top_section"]

        print("PDF에서 텍스트 추출 중...")
        full_text = extract_text(pdf_path)

        print("목차 기반 섹션 분할 중...")
        if selected_pattern_key == "f1_technical_regulations":
            # f1_technical_regulations 경우 2단계 조항까지 분리
            level1_pattern = selected_patterns["clause"]
            level2_pattern = selected_patterns["sub_clause"]
            sections = extract_sections_from_text2(full_text,
                                                   top_section_pattern_template=top_section_pattern,
                                                   level1_pattern_template=level1_pattern,
                                                   level2_pattern_template=level2_pattern)
        else:
            clause_pattern = selected_patterns["clause"]
            sections = extract_sections_from_text(full_text,
                                                  top_section_pattern=top_section_pattern,
                                                  clause_pattern_template=clause_pattern)

        print(f"총 {len(sections)}개 섹션 분할 완료.")

        all_sections.extend(sections)

        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        output_json_path = f"sections_{base_filename}.json"

        # (선택) 중간 결과 JSON 저장
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)

        print("문서 → LangChain Document 변환 중...")
        documents = create_documents(sections)

        print("Opensearch에 저장 중...")
        store_to_opensearch(documents, selected_pattern_key)

        print("저장 완료!")
