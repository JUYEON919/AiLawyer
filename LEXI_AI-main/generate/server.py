import os
import shutil
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from search import get_relevant_docs
from answer import generate_answer
from doc_create import create_contract_pdf, get_document_path  # ✅ 경로 함수 가져오기
import uvicorn
import requests

# ✅ 환경 변수 로드
load_dotenv()

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# ✅ CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ document/ 폴더 절대경로 설정 (프로젝트 루트)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # generate/ 폴더의 상위 폴더
DOCUMENT_DIR = os.path.join(BASE_DIR, "document")  # 프로젝트 루트의 document/ 폴더
os.makedirs(DOCUMENT_DIR, exist_ok=True)

# ✅ static/ 폴더 설정 (favicon.ico 제공)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ✅ FastAPI에서 정적 파일 제공 (문서 & 아이콘)
app.mount("/document", StaticFiles(directory=DOCUMENT_DIR), name="document")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ✅ 요청 받을 데이터 모델 정의
class QueryRequest(BaseModel):
    question: str

# ✅ SERVER_URL을 저장할 변수 (초기값은 None)
SERVER_URL = None

# ✅ SERVER_URL을 설정하는 새로운 엔드포인트 추가
@app.post("/set-server-url")
async def set_server_url(server_url: str):
    """Cloudtype 서버로부터 SERVER_URL을 받아서 설정"""
    global SERVER_URL
    SERVER_URL = server_url.strip()
    if not SERVER_URL.startswith(("http://", "https://")):
        SERVER_URL = "http://" + SERVER_URL
    logging.info(f"✅ SERVER_URL 설정됨: {SERVER_URL}")
    return {"message": "SERVER_URL이 성공적으로 설정되었습니다."}

@app.post("/gpu_ask")
async def gpu_ask(request: QueryRequest):
    """ ✅ 질문을 받아 관련 법률 검색 후, EXAONE 모델을 사용하여 답변 생성 """
    user_query = request.question.strip()
    logging.info(f"📝 질문 받음: {user_query}")

    if not user_query:
        logging.warning("⚠️ 빈 질문이 입력됨")
        return JSONResponse(content={"error": "질문을 입력하세요."}, status_code=400)

    try:
        # ✅ ChromaDB에서 관련 법률 및 판례 검색
        logging.info("🔍 관련 법률 및 판례 검색 시작")
        relevant_docs, sources, scores, law_numbers = get_relevant_docs(user_query)

        if not relevant_docs:
            logging.warning("⚠️ 참고할 법률 데이터 없음")
            return JSONResponse(content={"answer": "📌 참고할 법률 데이터를 찾을 수 없습니다.", "sources": []})

        logging.info("✅ 관련 법률 검색 완료")

        # ✅ EXAONE 모델을 이용한 답변 생성
        logging.info("🤖 EXAONE 모델을 사용하여 답변 생성 중...")
        response = generate_answer(user_query, relevant_docs, sources, scores)

        if not response:
            logging.error("❌ 모델이 빈 응답을 반환함")
            return JSONResponse(content={"error": "AI 모델이 유효한 답변을 생성하지 못했습니다."}, status_code=500)

        logging.info("✅ 답변 생성 완료")

        return JSONResponse(content={
            "answer": response,
            "sources": [
                {"law_number": law, "source": src, "score": sc}
                for law, src, sc in zip(law_numbers, sources, scores)
            ]
        })

    except Exception as e:
        logging.exception("❌ 서버 내부 오류 발생")
        return JSONResponse(content={"error": f"서버 내부 오류: {str(e)}"}, status_code=500)

# ✅ 문서 생성 모델 정의
class ContractRequest(BaseModel):
    contract_type: str
    party_a: str
    party_b: str
    contract_date: str
    additional_info: str = ""
    server_url: str  # main.py의 URL을 받기 위한 필드 추가

@app.post("/generate-document")
async def generate_contract(data: ContractRequest):
    """✅ 계약서를 생성하고 PDF 파일을 document/ 폴더에 저장"""
    logging.info(f"📝 문서 생성 요청 받음: {data}")

    try:
        # ✅ LOCAL_GPU_SERVER를 직접 전달받도록 수정
        server_url = data.server_url  # 여기서 data.server_url이 이미 LOCAL_GPU_SERVER 값임

        # ✅ PDF 생성
        pdf_path = create_contract_pdf(
            data.contract_type, data.party_a, data.party_b, data.contract_date, data.additional_info
        )
        file_name = os.path.basename(pdf_path)
        new_pdf_path = get_document_path(file_name)

        shutil.move(pdf_path, new_pdf_path)
        logging.info(f"✅ PDF 저장 완료: {new_pdf_path}")

        download_link = f"{server_url}/document/{file_name}"
        logging.info(f"🔗 다운로드 링크 생성 완료: {download_link}")

        return JSONResponse(content={"message": "문서가 생성되었습니다.", "download_link": download_link})

    except Exception as e:
        logging.exception("❌ 문서 생성 중 오류 발생")
        return JSONResponse(content={"error": f"서버 내부 오류: {str(e)}"}, status_code=500)

@app.get("/document/{file_name}")
async def download_file(file_name: str):
    """ ✅ 로컬 GPU 서버에서 PDF 다운로드 제공 """
    file_path = get_document_path(file_name)

    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "파일이 존재하지 않습니다."}, status_code=404)

    return FileResponse(file_path, media_type="application/pdf", filename=file_name)

# ✅ favicon.ico 제공
@app.get("/favicon.ico")
async def favicon():
    """ ✅ 브라우저에서 요청하는 favicon.ico 제공 """
    icon_path = os.path.join(STATIC_DIR, "favicon.ico")

    if not os.path.exists(icon_path):
        return JSONResponse(content={"error": "favicon.ico 파일이 존재하지 않습니다."}, status_code=404)

    return FileResponse(icon_path, media_type="image/x-icon")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)