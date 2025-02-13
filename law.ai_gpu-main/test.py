import torch

# GPU 사용 가능 여부 확인
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"GPU 개수: {torch.cuda.device_count()}")
print(torch.cuda.is_bf16_supported())  # True이면 사용 가능

# 사용 가능한 GPU 정보 출력
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - 메모리 할당 가능: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  - CUDA Compute Capability: {torch.cuda.get_device_capability(i)}")
else:
    print("❌ GPU를 사용할 수 없습니다.")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.randn(3, 3).to(device)
    print(f"✅ GPU에서 텐서 이동 성공! (장치: {device})")
except Exception as e:
    print(f"❌ GPU에서 텐서 이동 실패: {e}")
