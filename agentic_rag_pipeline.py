# react_agent: 내부에서 주어진 LLM과 tool들을 사용하여 에이전트를 생성
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

embeddings = OpenAIEmbeddings()

# OpenSearch 클라이언트 구성
vector_store = {
    "sporting_regulations": OpenSearchVectorSearch(
        index_name="f1_sporting_regulations",
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL + ":443",
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=60,
    ),
    "f1_pu_financial_regulations": OpenSearchVectorSearch(
        index_name="f1_pu_financial_regulations",
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL + ":443",
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=60,
    ),
    "f1_financial_regulations": OpenSearchVectorSearch(
        index_name="f1_financial_regulations",
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL + ":443",
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=60,
    ),
    "f1_technical_regulations": OpenSearchVectorSearch(
        index_name="f1_technical_regulations",
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL + ":443",
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )
}


# 각 Openseach 인덱스를 검색하는 함수 생성
def search_sporting_regulations(query: str) -> str:
    return vector_store["sporting_regulations"].similarity_search(query, k=5)


def search_f1_pu_financial_regulations(query: str):
    return vector_store["f1_pu_financial_regulations"].similarity_search(query, k=5)


def search_f1_financial_regulations(query: str):
    return vector_store["f1_financial_regulations"].similarity_search(query, k=5)


def search_f1_technical_regulations(query: str):
    return vector_store["f1_technical_regulations"].similarity_search(query, k=5)


tools = {
    Tool(
        name="search_sporting_regulations",
        func=search_sporting_regulations,
        description="MUST use for questions about race procedures, penalties, on-track rules, DRS, safety car, "
                    "formation laps, restart protocols, parc fermé conditions, driver conduct, and session-specific "
                    "regulations (e.g. sprint, qualifying, race).",
    ), Tool(
        name="search_f1_pu_financial_regulations",
        func=search_f1_pu_financial_regulations,
        description="MUST use for questions about power unit manufacturer budgets, cost cap limits, permitted and "
                    "excluded costs, financial reporting obligations, and FIA audit procedures.",
    ), Tool(
        name="search_f1_financial_regulations",
        func=search_f1_financial_regulations,
        description="MUST use for questions about car specifications, weight, tires, aerodynamics, chassis, "
                    "power unit allocations, fuel systems, ERS components, cooling systems, transmission, "
                    "and homologation requirements.",
    ), Tool(
        name="search_f1_technical_regulations",
        func=search_f1_technical_regulations,
        description="MUST use for questions about car design, materials, dimensions, weight distribution, power unit "
                    "configuration, energy recovery systems (ERS), fuel and lubrication, suspension, aerodynamics, "
                    "electronics, and homologation rules.",
    )
}


def build_agentic_rag():
    prompt_file_path = "txt files/Examples.txt"
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_str = f.read()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_str
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm, tools, prompt_template=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


response = build_agentic_rag().invoke({"input": input("Enter your question about F1 regulations: ")})
print("LLM 분석 결과:\n", response['output'])
