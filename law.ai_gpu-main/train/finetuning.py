import os
import json
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# ✅ GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 모델 및 토크나이저 로드
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ 법률 데이터 로드 함수 (연도별 폴더 포함)
def load_legal_data(json_folder):
    """ ✅ 연도별 폴더까지 순회하며 민사 판결문 데이터 로드 """
    data = []

    for year_folder in os.listdir(json_folder):
        year_path = os.path.join(json_folder, year_folder)
        if not os.path.isdir(year_path):
            continue  # 폴더가 아닌 경우 무시

        for file_name in os.listdir(year_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(year_path, file_name)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        case_data = json.load(f)

                    # ✅ JSON 필드 확인 및 기본값 설정
                    case_no = case_data.get("info", {}).get("caseNo", "사건번호 없음")
                    case_name = case_data.get("info", {}).get("caseNm", "사건명 없음")
                    related_laws = "\n".join(case_data.get("info", {}).get("relateLaword", ["관련 법률 없음"]))
                    facts = "\n".join(case_data.get("facts", {}).get("bsisFacts", ["사실 관계 없음"]))

                    # ✅ 학습 데이터로 사용할 문장 구성
                    text = f"사건명: {case_name}\n사건번호: {case_no}\n관련 법률: {related_laws}\n사실 관계: {facts}"
                    label = 1  # 단순 이진 분류

                    data.append((text, label))

                except json.JSONDecodeError:
                    print(f"❌ JSON 파일 파싱 오류: {file_name}")
                except Exception as e:
                    print(f"❌ 오류 발생: {file_name} - {e}")

    return data

# ✅ 데이터 경로 설정 (필터링된 법률 데이터 폴더)
TRAINING_PATH = "../법률, 규정/01.데이터/1.Training/라벨링데이터/TL_1.판결문/01.민사"
VALIDATION_PATH = "../법률, 규정/01.데이터/2.Validation/라벨링데이터/VL_1.판결문/01.민사"

# ✅ 데이터 로드
train_data = load_legal_data(TRAINING_PATH)
val_data = load_legal_data(VALIDATION_PATH)

print(f"📌 Training 데이터 개수: {len(train_data)}")
print(f"📌 Validation 데이터 개수: {len(val_data)}")

# ✅ 데이터셋 클래스 정의
class LegalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ✅ 데이터셋 준비
train_dataset = LegalDataset(train_data, tokenizer)
val_dataset = LegalDataset(val_data, tokenizer)

# ✅ 모델 로드 (이진 분류를 위한 헤드 추가)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

# ✅ 학습 파라미터 설정 (Colab 환경 최적화)
training_args = TrainingArguments(
    output_dir="./legal_bert_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=8,  # Colab GPU 최적화
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # GPU 메모리 최적화
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none"  # WandB 로그 비활성화
)

# ✅ Trainer 설정 (GPU 사용)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# ✅ 학습 시작
if __name__ == "__main__":
    print("📌 `legal-bert-base` 모델 Fine-tuning 시작...")

    # 데이터가 정상적으로 로드되었는지 확인
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("❌ Training 또는 Validation 데이터가 비어 있습니다. 데이터를 확인하세요.")

    trainer.train()

    # ✅ 모델 저장
    model.save_pretrained("./legal_bert_finetuned")
    tokenizer.save_pretrained("./legal_bert_finetuned")

    print("✅ Fine-tuning 완료! 모델이 `./legal_bert_finetuned` 폴더에 저장되었습니다.")
