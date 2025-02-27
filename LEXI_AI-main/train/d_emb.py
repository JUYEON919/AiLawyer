import os
import json
import chromadb
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ 데이터 경로 설정
BASE_DIR = "../법률, 규정/01.데이터"
TRAINING_PATH = os.path.join(BASE_DIR, "1.Training", "라벨링데이터", "TL_1.판결문")
VALIDATION_PATH = os.path.join(BASE_DIR, "2.Validation", "라벨링데이터", "VL_1.판결문")

# ✅ GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ ChromaDB 설정
CHROMA_DB_PATH = "../dataset/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
legal_cases_collection = chroma_client.get_or_create_collection(name="legal_cases")

# ✅ Fine-tuned `legal-bert-base` 모델 로드
MODEL_PATH = "../ft_legal_bert/checkpoint-1185"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
embedding_model = AutoModel.from_pretrained(MODEL_PATH).to(device)

# ✅ 불용어 리스트 (TF-IDF 활용)
STOPWORDS = set([
    "있다", "하는", "되어", "때문", "위해", "그리고", "따라서", "하지만", "대하여",
    "같은", "그러한", "경우", "그것", "이것", "저것", "이다", "였다", "되지", "없이",
    "하는", "하며", "라고", "까지", "하면서", "등등", "및", "이다"
])

def extract_keywords(text, top_n=5):
    """ ✅ TF-IDF를 활용하여 주요 키워드 추출 """
    if not text.strip():
        return "키워드 없음"
    
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=top_n)
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        return ", ".join(feature_names) if feature_names.size > 0 else "키워드 없음"
    except ValueError:
        return "키워드 없음"

def embed_text(text):
    """ ✅ Fine-tuned 모델을 사용하여 문장을 벡터화 (GPU 활용) """
    if not isinstance(text, str) or not text.strip():
        return None

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]

def extract_text_from_json(json_file):
    """ ✅ JSON 파일에서 법률 데이터 추출 """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        case_no = data["info"].get("caseNo", "").strip()
        case_name = data["info"].get("caseNm", "").strip()
        related_laws = "\n".join(data["info"].get("relateLaword", ["관련 법률 없음"]))
        disposal = "\n".join(data["disposal"].get("disposalcontent", ["처분 내용 없음"]))
        mentioned_items = "\n".join(data["mentionedItems"].get("rqestObjet", ["관련 청구 사항 없음"]))
        assrs = "\n".join(data["assrs"].get("acusrAssrs", ["주요 주장 없음"]))
        facts = "\n".join(data["facts"].get("bsisFacts", ["사실 관계 없음"]))
        dcss = "\n".join(data["dcss"].get("courtDcss", ["법원의 판단 없음"]))
        closing = "\n".join(data["close"].get("cnclsns", ["결론 없음"]))

        # ✅ 데이터 유효성 검사
        if not case_no or not case_name:
            return None, None, None, None

        text_data = f"""
        📌 사건번호: {case_no}
        📌 사건명: {case_name}
        📌 관련 법률: {related_laws}
        📌 처분 내용: {disposal}
        📌 관련 청구 사항: {mentioned_items}
        📌 주요 주장: {assrs}
        📌 사실 관계: {facts}
        📌 법원의 판단: {dcss}
        📌 결론: {closing}
        """

        # ✅ 키워드 추출 (빈 경우 "키워드 없음")
        keywords = extract_keywords(f"{related_laws} {disposal} {facts} {dcss}")

        return text_data.strip(), case_no, case_name, keywords
    except Exception as e:
        print(f"❌ JSON 파싱 오류: {json_file} - {e}")
        return None, None, None, None

def process_files(base_path, collection):
    """ ✅ 연도별 폴더까지 순회하여 JSON 파일을 벡터화 """
    for year_folder in ["1981~2016", "2017", "2018", "2019", "2020", "2021"]:
        case_path = os.path.join(base_path, "01.민사", year_folder)

        if not os.path.exists(case_path):
            print(f"⚠ 폴더 없음: {case_path}")
            continue

        for file_name in os.listdir(case_path):
            if not file_name.endswith(".json"):
                continue

            file_path = os.path.join(case_path, file_name)
            text_data, case_no, case_name, keywords = extract_text_from_json(file_path)

            if not text_data or not case_no or not case_name:
                print(f"⚠ 데이터 부족으로 건너뜀: {file_name}")
                continue

            embedding = embed_text(text_data)
            if embedding is None:
                print(f"⚠ 임베딩 실패로 건너뜀: 사건번호 {case_no}")
                continue

            # ✅ 중복 확인 후 데이터 추가
            existing_data = collection.get(ids=[case_no])
            if existing_data and existing_data["ids"]:
                print(f"🔄 기존 데이터 스킵: 사건번호 {case_no}")
                continue

            # ✅ 데이터 추가
            print(f"✅ 법률 데이터 추가 완료: 사건번호 {case_no}")
            collection.add(
                ids=[case_no],
                embeddings=[embedding],
                metadatas=[{
                    "case_number": case_no,
                    "title": case_name,
                    "keywords": keywords,
                    "text": text_data
                }]
            )

# ✅ 실행
if __name__ == "__main__":
    print("📌 Training 데이터 벡터화 시작...")
    process_files(TRAINING_PATH, legal_cases_collection)

    print("📌 Validation 데이터 벡터화 시작...")
    process_files(VALIDATION_PATH, legal_cases_collection)

    print("✅ 모든 법률 데이터 벡터화 완료!")
