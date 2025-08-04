# react_agent: 내부에서 주어진 LLM과 tool들을 사용하여 에이전트를 생성
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

embeddings = OpenAIEmbeddings()

# OpenSearch 클라이언트 구성
vector_store = OpenSearchVectorSearch(
    embedding_function=embeddings,
    opensearch_url=OPENSEARCH_URL + ":443",
    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection,
    timeout=60,
)


# 각 Openseach 인덱스를 검색하는 함수 생성
def search_sporting_regulations(query: str) -> str:
    index_name = "f1_sporting_regulations"
    results = vector_store.similarity_search(query, k=5, index_name=index_name)

    return results


def search_f1_pu_financial_regulations(query: str):
    index_name = "f1_pu_financial_regulations"
    results = vector_store.similarity_search(query, k=5, index_name=index_name)

    return results


def search_f1_financial_regulations(query: str):
    index_name = "f1_financial_regulations"
    results = vector_store.similarity_search(query, k=5, index_name=index_name)

    return results


def search_f1_technical_regulations(query: str):
    index_name = "f1_technical_regulations"
    results = vector_store.similarity_search(query, k=5, index_name=index_name)

    return results

