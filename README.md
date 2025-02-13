# ** AiLawyer - Ai 법률 상담 채팅 웹 어플리케이션**  


> **AiLawyer**는 LLM 기반의 법률 상담 챗봇으로, 사용자가 법적 문제에 대한 질문을 하면 적절한 법률 조항과 판례를 바탕으로 답변을 제공하는 웹 애플리케이션입니다.  

---

## 🛠 Development Environment  

### 🌐 Frontend  
- **HTML/CSS**: 웹페이지 구조 및 스타일링  
- **JavaScript**: 동적 기능 및 클라이언트 측 로직 구현  
- **React**: 컴포넌트 기반 UI/UX 구현  

### 🖥 Backend  
- **Spring Boot 3.0+**: 웹 프레임워크  
- **Java (JDK 17)**: 비즈니스 로직 구현  
- **Python 3.12.8**: 머신러닝 및 딥러닝 예측 모델 구현  
- **FastAPI 3.10.6**: AI 모델 호출 및 결과 제공 (RESTful API)  

### 🤖 AI & LLM  
- **Exaone 3.5**: 대규모 언어 모델을 통한 AI 서비스  

### 🗄 Database  
- **MariaDB / MySQL (10.6)**: 데이터 저장소 (RDBMS)  
- **MyBatis 3.5.9**: 데이터베이스 연동 및 쿼리 처리  

---

## **🔧 설치 및 실행 방법**  

### **🔢 프로젝트 클론**  
```bash
git clone https://github.com/JUYEON919/law.ai_gpu.git
cd law.ai_gpu
```

---

## **🚀 실행 방법 (GPU / CPU 선택 가능)**  

### ✅ **GPU 기반 실행 (NVIDIA CUDA 사용)**  
> **💡 NVIDIA GPU가 재장되어 있는 환경에서 실행하는 경우**  

#### **🔢 CUDA 및 PyTorch 설치**  
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
> 💡 `cu118`은 CUDA 11.8을 의미하며, 사용자의 CUDA 버전에 맞게 변경해야 합니다.

#### **🔢 프로젝트 패키지 설치**  
```bash
pip install -r requirements.txt
```

#### **🔢 FastAPI 서버 실행**  
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### **🔢 Streamlit 프론트엔드 실행**  
```bash
streamlit run web/app.py
```

---

### ✅ **CPU 기반 실행 (GPU 없이 실행 가능)**  
> **💡 NVIDIA GPU가 없는 환경에서 실행하는 경우**  

#### **🔢 CPU 전용 PyTorch 설치**  
```bash
pip install torch torchvision torchaudio
```

#### **🔢 프로젝트 패키지 설치**  
```bash
pip install -r requirements.txt
```

#### **🔢 FastAPI 서버 실행**  
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### **🔢 Streamlit 프론트엔드 실행**  
```bash
streamlit run web/app.py
```

---



---
## ⚠️ 주의할점

- 프로젝트 실행 전, `data` 폴더 위치설정, 폴더 내 데이터 파일이 존재하는지 확인하세요.
- 프로젝트 실행 시, 현재 경로를 프로젝트의 루트 디렉토리로 설정해야 합니다.
- 일부 머신러닝 모델 학습에는 **수 분**이 소요될 수 있습니다.


