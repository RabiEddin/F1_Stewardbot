from fastapi import FastAPI, UploadFile, File, Form, Depends, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.globals import set_verbose
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
import cv2
import base64
from openai import OpenAI
from PIL import Image
import io
import tempfile

from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from redis import asyncio as aioredis

set_verbose(True)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_URL = "search-f1--stewardbot-vk5ukbpmlhss2ef3jgnwwervla.us-east-2.es.amazonaws.com"
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

app = FastAPI()

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Static files setup
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Load users from a JSON file
def load_users():
    if not os.path.exists('users.json'):
        return {}
    with open('users.json', 'r') as f:
        return json.load(f)


@app.on_event("startup") # Initialize FastAPILimiter with Redis on startup
async def startup():
    redis = aioredis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis)


@app.exception_handler(HTTP_429_TOO_MANY_REQUESTS)
async def rate_limit_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=HTTP_429_TOO_MANY_REQUESTS, content={"message": "Too many requests"})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if not request.session.get('user'):
        return RedirectResponse(url="/login")
    with open("app/static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("app/static/login.html", "r") as f:
        return HTMLResponse(content=f.read())


class User(BaseModel):
    username: str
    password: str


@app.post("/login")
async def login(user: User, request: Request):
    users = load_users()
    if user.username in users and pwd_context.verify(user.password, users[user.username]):
        request.session['user'] = user.username
        return JSONResponse(content={"message": "Login successful"})
    return JSONResponse(status_code=401, content={"message": "Invalid credentials"})


@app.get("/logout")  # http get 방식으로 아래 logout 함수를 호출하면 세션에서 user를 제거하고 로그인 페이지로 리다이렉트합니다.
async def logout(request: Request):
    request.session.pop('user', None)  # 세션에서 'user' 키를 제거합니다.
    return RedirectResponse(url="/login")  # 로그인 페이지로 리다이렉트합니다.


class SituationRequest(BaseModel):
    situation: str


def connect_to_vectorstore():
    embeddings = OpenAIEmbeddings()

    vector_store = OpenSearchVectorSearch(
        index_name="f1_rules",
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL + ":443",
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=120,
    )
    return vector_store


def search_related_rules(user_input, vector_store, k=5):
    results = vector_store.similarity_search(user_input, k=k)
    return results


def build_reasoning_chain():
    prompt_file_path = "txt files/Examples.txt"
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_str = f.read()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_str
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


# Video processing functions
client = OpenAI(api_key=OPENAI_API_KEY)


def extract_frames_from_video(video_path, fps=2, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate / fps)
    frames = []
    frame_count = 0

    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            frames.append(img_b64)
        frame_count += 1

    cap.release()
    return frames


def describe_frame_b64(frames, prompt_file_path, date, race_country):
    with open(prompt_file_path, 'r', encoding='utf-8') as pf:
        prompt_text = pf.read().strip()

    vision_input = [
                       {"type": "text",
                        "text": prompt_text + f"\ndate: {date}\nRace country: {race_country}"}
                   ] + [
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                       for img_b64 in frames
                   ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": vision_input}]
    )
    return response.choices[0].message.content


def build_situation_from_video(video_path, date, race_country):
    frames = extract_frames_from_video(video_path)
    descriptions = describe_frame_b64(frames, "txt files/situation_description.txt", date, race_country)
    return descriptions


@app.post("/predict", dependencies=[Depends(RateLimiter(times=5, seconds=60))]) # Rate limiting to 5 requests per minute
async def predict_violation(request: Request, situation_request: SituationRequest):
    if not request.session.get('user'): # Check if user is authenticated, raise 예외 발생
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        analysis_result = run_rag_pipeline(situation_request.situation)
        return {"analysis": analysis_result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze_video", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def analyze_video(request: Request, video: UploadFile = File(...), date: str = Form(...),
                        race_country: str = Form(...)):
    if not request.session.get('user'):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            contents = await video.read()
            tmp_file.write(contents)
            tmp_video_path = tmp_file.name

        # Process the video to get situation description
        situation_description = build_situation_from_video(tmp_video_path, date, race_country)

        # Run RAG pipeline with the situation description
        analysis_result = run_rag_pipeline(situation_description)

        # Clean up the temporary file
        os.unlink(tmp_video_path)

        return {"analysis": analysis_result}
    except Exception as e:
        return {"error": str(e)}
