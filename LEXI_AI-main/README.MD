# 프로젝트 설정 및 실행 방법

## 1. 가상환경 설정
```bash
python -m venv venv  # 가상환경 생성
venv/Scripts/activate  # 가상환경 활성화 (Windows)
source venv/bin/activate  # (Mac/Linux)
```

## 2. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

## 3. 학습 데이터 준비
```bash
cd train  # train 폴더로 이동
```

### 📌 CPU 사용 시 (시간 소요 많음)
1. ChromaDB 다운로드 후, `dataset/chroma_db/` 폴더에 학습된 모델 배치  
2. Hugging Face에서 파인튜닝된 `lawyer-ai/law_ko_ft_legal_bert` 모델 사용  
3. `server.py`에서 PyTorch의 BFloat16 데이터를 float32로 변환 후 numpy로 변환하여 사용  

### 🚀 GPU 사용 시 (빠른 실행)
```bash
python d_emb  # 판례 목록 데이터 임베딩
python ldata_emb  # 데이터 임베딩
python finetuning  # 모델 파인튜닝
python check_chroma.py  # 데이터 로드 후 확인
```

## 4. 서버 실행
```bash
cd ..  # 상위 폴더로 이동
cd generate  # generate 폴더로 이동
uvicorn server:app --reload  # FastAPI 서버 실행
```
