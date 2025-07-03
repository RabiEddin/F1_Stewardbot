import os
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.globals import set_verbose

set_verbose(True)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")


def connect_to_vectorstore():
    embeddings = OpenAIEmbeddings()

    vector_store = OpenSearchVectorSearch(
        index_name="f1_rules",
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL+":443",
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )
    return vector_store


def search_related_rules(user_input, vector_store, k=5):
    results = vector_store.similarity_search(user_input, k=k)
    return results


def build_reasoning_chain():
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an F1 regulations expert. Given the rule context below, and a situation described by the user, explain whether a rule violation occurred and why. And answer in the language of the question.

RULE CONTEXT:
{context}

SITUATION:
{question}

EXPERT ANALYSIS:
"""
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain


def run_rag_pipeline(user_input):
    vector_store = connect_to_vectorstore()
    docs = search_related_rules(user_input, vector_store)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = build_reasoning_chain()
    result = chain.run({"context": context, "question": user_input})
    return result


situation = "랜도 노리스가 점프 스타트를 했어. 패널티가 어떻게 나올까?"
answer = run_rag_pipeline(situation)
print("LLM 분석 결과:\n", answer)
