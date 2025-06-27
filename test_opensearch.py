import os
from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_community.vectorstores import OpenSearchVectorSearch

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

embeddings = OpenAIEmbeddings()

index_name = "f1_rules"

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
    opensearch_url=OPENSEARCH_URL+":443",
    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection,
    timeout=60 # 업로드 안되는 이유: 기본 디폴트 timeout이 짧게 되어 있어서 -> 해결: timeout 늘림
)

retriever = vector_store.as_retriever()
retriever.invoke("f1에서 패널티를 받는 주된 상황은?")

