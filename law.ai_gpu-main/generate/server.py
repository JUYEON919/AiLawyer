import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from search import get_relevant_docs
from answer import generate_answer
import uvicorn

# ✅ .env 파일 로드
load_dotenv()

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log"),  # 로그를 파일에 저장
        logging.StreamHandler()  # 콘솔에도 출력
    ]
)

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 모든 도메인 허용 (보안 문제 있으면 특정 도메인만)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 요청 받을 데이터 모델 정의
class QueryRequest(BaseModel):
    question: str

@app.post("/gpu_ask")
async def gpu_ask(request: QueryRequest):
    """ ✅ 질문을 받아 관련 법률 검색 후, EXAONE 모델을 사용하여 답변 생성 """
    user_query = request.question.strip()
    logging.info(f"📝 질문 받음: {user_query}")

    if not user_query:
        logging.warning("⚠️ 빈 질문이 입력됨")
        return {"error": "질문을 입력하세요."}

    try:
        # ✅ ChromaDB에서 관련 법률 및 판례 검색
        logging.info("🔍 관련 법률 및 판례 검색 시작")
        relevant_docs, sources, scores, law_numbers = get_relevant_docs(user_query)

        if not relevant_docs:
            logging.warning("⚠️ 참고할 법률 데이터 없음")
            return {"answer": "📌 참고할 법률 데이터를 찾을 수 없습니다.", "sources": []}

        logging.info("✅ 관련 법률 검색 완료")

        # ✅ EXAONE 모델을 이용한 답변 생성
        logging.info("🤖 EXAONE 모델을 사용하여 답변 생성 중...")
        response = generate_answer(user_query, relevant_docs, sources, scores)
        logging.info("✅ 답변 생성 완료")

        return {
            "answer": response,
            "sources": [
                {"law_number": law, "source": src, "score": sc}
                for law, src, sc in zip(law_numbers, sources, scores)
            ]
        }

    except Exception as e:
        logging.exception("❌ 서버 내부 오류 발생")
        return {"error": f"서버 내부 오류: {str(e)}"}

# ✅ 서버 상태 확인을 위한 엔드포인트 추가
@app.get("/health")
def health_check():
    logging.info("✅ Health Check 요청 받음")
    return {"status": "OK", "message": "GPU Server is running"}

# ✅ 로컬 GPU 서버 실행
if __name__ == "__main__":
    logging.info("🚀 FastAPI GPU 서버 시작됨")
    uvicorn.run(app, host="0.0.0.0", port=8001)  # 외부 요청 허용
