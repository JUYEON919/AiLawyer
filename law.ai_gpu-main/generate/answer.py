from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ✅ EXAONE 모델 로드 설정
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 모델 로드 (VRAM 절약을 위해 `torch_dtype=torch.float16` 사용)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)

def generate_answer(query, relevant_docs, sources, scores):
    """ ✅ EXAONE 모델을 이용하여 법률 답변 생성 (GPU 활용) """

    if not relevant_docs:
        relevant_docs = ["📌 참고할 법률 조항을 찾을 수 없습니다. 일반적인 법률 원칙을 적용하세요."]

    # ✅ `scores`가 2차원 리스트(`list[list]`)인지 확인 후 1차원으로 변환
    scores = [s for sublist in scores for s in (sublist if isinstance(sublist, list) else [sublist])]

    try:
        scores = [float(s) for s in scores]
    except ValueError:
        scores = [0.0] * len(scores)

    # ✅ 토큰 수 초과 방지: 관련 법률 내용을 3000자까지만 사용
    relevant_text = "\n".join(relevant_docs)[:3000]

    prompt = f"""
    [사용자 질문]
    당신은 대한민국 변호사입니다.
    그리고 모든 답변은 한국어로 해주세요.
    고객의 질문을 세밀하게 분석하여, 전문적인 답변을 사용자에게 제공해주세요.
    주로 아래에 관련 법률 및 판례를 위주로 답변을 생성하세요.
    {query}

    [관련 법률 및 판례]
    {relevant_text}

    [변호사 답변]
    """

    # ✅ 너무 긴 입력 방지: 4096 토큰 제한
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,  # ✅ 출력 길이를 512 토큰으로 제한
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True).split("[변호사 답변]")[-1].strip()

    return answer
