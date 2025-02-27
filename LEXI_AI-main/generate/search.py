import os
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re

# ✅ ChromaDB 설정
CHROMA_DB_PATH = "../dataset/chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
legal_cases_collection = chroma_client.get_or_create_collection(name="legal_cases")  # 법률 데이터
legal_precedents_collection = chroma_client.get_or_create_collection(name="legal_precedents")  # 판례 데이터

# ✅ Fine-tuned `legal-bert-base` 모델 로드
MODEL_PATH = "../ft_legal_bert/checkpoint-1185"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
embedding_model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)

def embed_text(text):
    """ ✅ Fine-tuned 모델을 사용하여 문장을 벡터화 (GPU 활용) """

    if text is None or not isinstance(text, str) or not text.strip():
        raise ValueError("❌ 오류: `text` 값이 유효하지 않습니다.")

    text = re.sub(r"\s+", " ", text.strip())  # 여러 개 공백 제거

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

    return embedding.cpu().numpy().flatten().tolist()

def get_relevant_docs(query, top_k=5):
    """ ✅ 사용자의 질문을 벡터화하고, 관련 법률 및 판례 데이터를 검색 """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("❌ 오류: 입력된 질문이 유효하지 않습니다.")

    query_embedding = embed_text(query)

    results_cases = legal_cases_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    results_precedents = legal_precedents_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    relevant_texts, sources, scores, law_numbers = [], [], [], []

    def process_results(results):
        # ✅ 검색된 결과 가공
        # ✅ 검색된 결과 가공
        for i, meta_list in enumerate(results.get("metadatas", [])):
            for meta in meta_list:
                text = meta.get("text", results["documents"][i] if results.get("documents") else None)
                score = results.get("distances", [])[i] if results.get("distances") else 0.0

                # ✅ "사건번호"를 올바르게 가져오기 (법률 & 판례 구분)
                case_no = meta.get("case_no") or meta.get("사건번호") or meta.get("law_number") or "사건번호 없음"

                # ✅ 사건번호가 `text` 내에 포함되어 있다면 가져오기 (예외 처리)
                if case_no == "사건번호 없음" and text:
                    match = re.search(r"📌 사건번호: ([\w\d-]+)", text)
                    if match:
                        case_no = match.group(1)

                # ✅ 불필요한 공백 정리
                if isinstance(text, list):
                    text = "\n".join([t if t is not None else "" for t in text])

                if text and text.strip():
                    relevant_texts.append(text)
                    sources.append(case_no)  # ✅ 사건번호를 출처 정보로 사용
                    scores.append(score)
                    law_numbers.append(case_no)  # ✅ 사건번호를 법률 번호로 저장



    process_results(results_cases)
    process_results(results_precedents)

    # ✅ 2차원 리스트(`list[list[float]]`)인 경우 1차원 리스트로 변환
    scores = [s for sublist in scores for s in (sublist if isinstance(sublist, list) else [sublist])]

    return relevant_texts, sources, scores, law_numbers
