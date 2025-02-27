가상환경 설치 (python -m venv venv) (venv/Scripts/activate으로 가상환경 실행)

pip install -r requirements.txt로 라이브러리 설치

cd train 으로 train폴더로 이동

## cpu사용시 (소용시간이 많이 소요)

chromadb다운후 dataset/chroma_db/안에 학습된 모델넣어두기

허깅페이스에 파인튜닝완료해서 넣어둔 lawyer-ai/law_ko_ft_legal_bert 모델사용 server.py에서 PyTorch에서 BFloat16 데이터를 float32로 변환한 후 numpy로 변환

## gpu사용시

python d_emb 을 실행하여 판례 목록 데이터 임베딩 python ldata_emb 을 실행하여 데이터 임베딩 python finetuning 을 실행하여 파인튜닝

check_chroma.py로 데이터 로드 후 확인.

cd .. 으로 폴더를 나오고 cd generate으로 경로 지정 uvicorn server:app --reload 실행
