import numpy as np
import os

print("=== 파일 로드 진단 ===\n")

# NPZ 파일 확인
print("1. NPZ 파일 목록:")
for f in os.listdir('.'):
    if f.endswith('.npz'):
        print(f"   - {f}")

# 로드 테스트
print("\n2. 로드 테스트:")
try:
    data = np.load('marine_seismic_data.npz', allow_pickle=True)
    print(f"   ✓ marine_seismic_data.npz 로드 성공")
    print(f"   Keys: {list(data.keys())}")
    
    # 모델 로드
    model = data['model'].item()
    print(f"   ✓ 모델 로드 성공")
    print(f"   지층 수: {len(model['name'])}")
    
except FileNotFoundError:
    print("   ✗ 파일이 없습니다. 먼저 시뮬레이션을 실행하세요:")
    print("      python marine_seismic_simulation.py")
except Exception as e:
    print(f"   ✗ 오류: {e}")
