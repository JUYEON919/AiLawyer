import chromadb

# ✅ ChromaDB 연결
CHROMA_DB_PATH = "../dataset/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# ✅ 컬렉션 불러오기
try:
    legal_cases = chroma_client.get_collection(name="legal_cases")  # 법률 데이터
    legal_precedents = chroma_client.get_collection(name="legal_precedents")  # 판례 목록 데이터
except (chromadb.errors.InvalidCollectionException, KeyError):
    print("❌ ChromaDB 컬렉션을 찾을 수 없습니다. 먼저 `d_emb.py` 및 `ld_emb.py`를 실행하세요.")
    exit()

# ✅ 데이터 개수 확인
legal_cases_count = legal_cases.count()
legal_precedents_count = legal_precedents.count()

print(f"📌 저장된 법률 데이터 개수: {legal_cases_count}")
print(f"📌 저장된 판례 데이터 개수: {legal_precedents_count}")

# ✅ 샘플 데이터 가져오기 (각 컬렉션에서 3개씩 출력)
sample_cases = legal_cases.get(include=["metadatas"], limit=3)
sample_precedents = legal_precedents.get(include=["metadatas"], limit=3)

# ✅ 데이터 출력 함수
def print_sample_data(sample_data, data_type):
    metadata_list = sample_data.get("metadatas", [{}])  # 🔥 빈 리스트 방지
    if metadata_list and metadata_list[0]:  # 🔥 데이터가 존재하는지 확인
        print(f"\n📌 저장된 {data_type} 데이터 샘플 (최대 3개):")
        for i, metadata in enumerate(metadata_list):
            case_no = metadata.get("case_number", metadata.get("law_number", "⚠ 사건번호 없음"))
            title = metadata.get("제목", "⚠ 제목 없음")
            content = metadata.get("text", "⚠ 내용 없음").split("\n")[0][:200]  # ✅ 첫 번째 줄만 200자 출력
            source = metadata.get("source", "출처 정보 없음")

            print(f"\n🔹 샘플 {i+1}:")
            print(f"📌 사건번호: {case_no}")
            print(f"📖 제목: {title}")
            print(f"📖 내용 미리보기: {content}...")
            print(f"📖 출처: {source}")
    else:
        print(f"⚠ 저장된 {data_type} 데이터가 없습니다.")

# ✅ 샘플 데이터 출력
print_sample_data(sample_cases, "법률")
print_sample_data(sample_precedents, "판례")
