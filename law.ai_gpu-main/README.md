1. 가상환경 설치 (python -m venv venv) (venv/Scripts/activate으로 가상환경 실행)
2. pip install -r requirements.txt로 라이브러리 설치
3. cd train 으로 train폴더로 이동
4. data_embedding -> 판례목록.json / law_data_embedding.py -> 법률 데이터 / regulation_data_embedding.py -> 약관
5. python data_embedding 을 실행하여 판례 목록 데이터 임베딩
6. python law_data_embedding 을 실행하여 데이터 임베딩
7. python regulation_data_embedding 을 실행하여 데이터 임베딩
8. check_chroma.py로 데이터 로드 후 확인.
9. cd .. 으로 폴더를 나오고 cd generate으로 경로 지정
10. python generate.py로 실행.