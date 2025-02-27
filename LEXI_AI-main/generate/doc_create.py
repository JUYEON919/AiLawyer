import os
import uuid
import pdfkit
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ 프로젝트 루트의 `document/` 폴더를 사용하도록 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # `generate/` 폴더의 상위 폴더
DOCUMENT_PATH = os.path.join(BASE_DIR, "document")

# ✅ `document/` 폴더가 없으면 생성
os.makedirs(DOCUMENT_PATH, exist_ok=True)

def get_document_path(filename):
    """✅ 생성된 PDF의 전체 경로 반환"""
    return os.path.join(DOCUMENT_PATH, filename)

MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# ✅ 필요한 패키지 설치 확인
try:
    from accelerate import infer_auto_device_map
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
except ImportError:
    raise ImportError("🚨 다음 패키지들이 필요합니다:\n"
                     "pip install accelerate bitsandbytes")

# ✅ 4비트 양자화 설정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 4비트 양자화 사용
    bnb_4bit_compute_dtype=torch.float16,  # 연산 시 float16 사용
    bnb_4bit_use_double_quant=True,   # 이중 양자화로 메모리 추가 절약
    bnb_4bit_quant_type="nf4",        # NF4 양자화 타입 사용
)

# ✅ GPU 사용 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 모델 로드 (4비트 양자화 + 메모리 최적화)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config,  # 4비트 양자화 설정 적용
    torch_dtype=torch.float16,  # float16으로 데이터 타입 설정
)

def generate_contract_text(contract_type, party_a, party_b, contract_date, additional_info=""):
    """ ✅ EXAONE 모델을 이용하여 계약서 생성 (프롬프트 제거 + 불필요한 텍스트 필터링) """
    logging.info(f"🔍 EXAONE에 문서 요청: {contract_type}, {party_a}, {party_b}, {contract_date}")

        # 문서 유형에 따른 특화된 지시사항 설정
    document_type_instructions = {
        "계약서": "계약 조항은 명확하고 구체적으로 작성하며, 각 조항에 번호를 매겨주세요.",
        "합의서": "당사자 간의 합의 사항을 구체적으로 나열하고, 합의 조건과 이행 방법을 명시하세요.",
        "진술서": "진술인의 진술 내용을 시간 순서대로 구체적으로 작성하고, 사실관계를 명확히 기술하세요."
    }

    prompt = f"""
    다음 정보를 바탕으로 {contract_type}를 작성해주세요.
    
    [필수 작성 지침]
    1. 문서 형식:
       - 제목, 전문, 본문, 날짜, 서명란 순으로 구성
       - 각 조항은 명확한 번호와 소제목 포함
       - 전문에는 계약의 배경과 목적을 명시
    
    2. 문서 특성:
       {document_type_instructions.get(contract_type, "계약 조항을 명확하게 작성하세요.")}
    
    3. 형식 요구사항:
       - 법률 용어는 정확하고 일관되게 사용
       - 모든 금액과 날짜는 숫자와 한글을 병기
       - 각 페이지 하단에 당사자 날인란 포함
    
    [계약 정보]
    계약 당사자:
    - 갑측: {party_a}
    - 을측: {party_b}
    
    계약일자: {contract_date}
    
    추가 계약내용:
    {additional_info}

    --- 계약서 시작 ---
    """

    # ✅ 모델 입력값 설정
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=800,  
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # ✅ 생성된 계약서 텍스트 정리
    contract_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # ✅ 계약서 본문이 시작되는 위치 찾기
    start_marker = "--- 계약서 시작 ---"
    start_index = contract_text.find(start_marker)

    if start_index != -1:
        contract_text = contract_text[start_index + len(start_marker):].strip()  # 🚀 계약서 시작 지점부터만 유지

    # ✅ 불필요한 문구 제거
    unwanted_phrases = [
        "계약서 본문", "계약서 종료",  # 🚀 불필요한 텍스트 제거
        "반드시", "포함하지 마세요", "위 정보를 기반으로", "계약서 본문만 반환하세요"
    ]
    for phrase in unwanted_phrases:
        contract_text = contract_text.replace(phrase, "").strip()

    logging.info("✅ 문서 생성 완료 (프롬프트 및 불필요한 텍스트 제거 완료)")
    return contract_text

def create_contract_pdf(contract_type, party_a, party_b, contract_date, additional_info=""):
    """
    ✅ Exaone 모델을 사용하여 계약서 내용을 생성하고 PDF로 저장
    """
    logging.info(f"📄 문서 생성 시작: {contract_type}, {party_a}, {party_b}, {contract_date}")

    # ✅ Exaone을 사용하여 계약서 텍스트 생성
    contract_text = generate_contract_text(contract_type, party_a, party_b, contract_date, additional_info)

    # ✅ PDF 파일명 생성 (파일명에 날짜와 계약 유형 포함)
    file_name = f"{contract_type}_{party_a}_{party_b}_{contract_date.replace('-', '')}_{uuid.uuid4().hex}.pdf"
    pdf_path = get_document_path(file_name)  # ✅ `document/` 폴더에 저장

    # ✅ PDF 변환
    convert_to_pdf(contract_text, pdf_path)

    logging.info(f"✅ PDF 문서 저장 완료: {pdf_path}")

    return pdf_path

def convert_to_pdf(text, pdf_path):
    """
    ✅ HTML 기반 PDF 변환 (한글 폰트 적용)
    """
    html_content = f"""
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        @font-face {{
          font-family: 'Nanum Gothic';
          src: local('NanumGothic'), url('https://fonts.gstatic.com/ea/nanumgothic/v5/NanumGothic-Regular.ttf') format('truetype');
        }}
        body {{
          font-family: 'Nanum Gothic', 'Malgun Gothic', sans-serif;
          line-height: 1.6;
          margin: 20px;
          padding: 20px;
          border: 2px solid #000;
          max-width: 800px;
        }}
        h1 {{
          text-align: center;
          text-decoration: underline;
        }}
        .contract {{
          margin-top: 20px;
          padding: 20px;
          border: 1px solid #ddd;
          background: #f9f9f9;
        }}
        .signature {{
          margin-top: 40px;
          text-align: left;
          padding-left: 20px;
        }}
      </style>
    </head>
    <body>
      <h1>{text.splitlines()[0]}</h1>
      <div class="contract">
        <p>{text.replace("\n", "<br>")}</p>
      </div>
    </body>
    </html>
    """

    # ✅ wkhtmltopdf 실행 경로 명확히 지정 (Cloudtype 환경에 맞춰 설정)
    config = pdfkit.configuration(wkhtmltopdf="/usr/bin/wkhtmltopdf")
    
    # ✅ PDF 생성 시 config 적용
    pdfkit.from_string(html_content, pdf_path, configuration=config)
