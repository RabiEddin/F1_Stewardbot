import fitz
import re
import os
import json
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import ElasticsearchStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text(pdf_path): # PDF 텍스트 추출
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_sections_from_text(text): # 목차 기반 섹션 분할
    # 상위 섹션: 예) 1) General Principles
    top_section_pattern = r"(?m)^(?:\s*)?(\d+)\)\s+([A-Z][^\n]+)"  # ex: 4) LICENCES
    top_sections = list(re.finditer(top_section_pattern, text))

    result = []

    for i, top in enumerate(top_sections):
        section_num = top.group(1).strip()
        section_title = top.group(2).strip()
        start = top.end()
        end = top_sections[i + 1].start() if i + 1 < len(top_sections) else len(text)

        section_text = text[start:end]

        # 하위 조항 분리: ex. 4.1, 4.2, ...
        clause_pattern = rf"(?m)^\s*({section_num}\.\d+)\s+[A-Z]" # 4.1 + 대문자로 시작하는 경우만 패턴으로 인식
        clause_matches = list(re.finditer(clause_pattern, section_text))

        if clause_matches:
            for j, clause in enumerate(clause_matches):
                clause_num = clause.group(1).strip()
                clause_start = clause.end()
                clause_end = clause_matches[j + 1].start() if j + 1 < len(clause_matches) else len(section_text)

                content = section_text[clause_start-1:clause_end].strip() # -1인유 패턴에 첫번쨰 대문자가 들어가 있어 -1로 조정

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

# def seperate_chunck_from_test(text): # 보류: 일반적인 청크 사이즈로 나누기

def create_documents(sections): # LangChain Document 생성
    return [
        Document(
            page_content=sec["content"],
            metadata={"section": sec["section"], "sub_section": sec["sub_section"], "title": sec["title"]}
        )
        for sec in sections
    ]

def store_to_elasticsearch(documents): # Elasticsarch에 저장
    embeddings = OllamaEmbeddings(model="llama3.1")
    es_store = ElasticsearchStore.from_documents(
        documents,
        embedding=embeddings,
        es_url=os.getenv("ELASTICSEARCH_URL"),
        index_name="f1_rules"
    )
    return es_store

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

    print("Elasticsearch에 저장 중...")
    store_to_elasticsearch(documents)

    print("저장 완료!")