#!/usr/bin/env python3
"""
문제 해결 가이드 (Troubleshooting Guide)
파일 로드 관련 문제 해결 방법
"""

import numpy as np
import sys
import os

def check_file_exists(filename):
    """파일 존재 여부 확인"""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"✓ {filename} 파일 존재 (크기: {size:,} bytes)")
        return True
    else:
        print(f"✗ {filename} 파일 없음")
        return False

def load_npz_safely(filename):
    """안전한 NPZ 파일 로드"""
    print(f"\n[{filename} 로드 시도]")
    
    # 1. 파일 존재 확인
    if not check_file_exists(filename):
        print(f"  해결방법: 먼저 시뮬레이션을 실행하여 데이터를 생성하세요")
        print(f"  예: python marine_seismic_simulation.py")
        return None
    
    # 2. 파일 로드
    try:
        data = np.load(filename, allow_pickle=True)
        print(f"✓ 파일 로드 성공")
        print(f"  포함된 키: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"✗ 로드 실패: {type(e).__name__}")
        print(f"  오류 메시지: {e}")
        
        # 일반적인 오류별 해결방법
        if "corrupted" in str(e).lower() or "invalid" in str(e).lower():
            print(f"  해결방법: 파일이 손상되었을 수 있습니다. 재생성하세요")
            print(f"  예: python marine_seismic_simulation.py")
        elif "permission" in str(e).lower():
            print(f"  해결방법: 파일 접근 권한을 확인하세요")
            print(f"  예: chmod 644 {filename}")
        else:
            print(f"  해결방법: 파일을 삭제하고 재생성하세요")
        
        return None

def load_model_from_npz(data):
    """NPZ에서 모델 딕셔너리 로드"""
    try:
        model = data['model'].item()
        print(f"✓ 모델 로드 성공")
        print(f"  타입: {type(model)}")
        if isinstance(model, dict):
            print(f"  키: {list(model.keys())}")
        return model
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        print(f"  해결방법: allow_pickle=True 옵션을 사용하세요")
        print(f"  예: np.load(filename, allow_pickle=True)")
        return None

def example_usage():
    """올바른 사용 예제"""
    print("\n" + "="*70)
    print("올바른 파일 로드 예제")
    print("="*70)
    
    code_example = '''
# 예제 1: 기본 로드
import numpy as np

data = np.load('marine_seismic_data.npz', allow_pickle=True)
time = data['time']
clean = data['clean']
noisy = data['noisy']
model = data['model'].item()  # 딕셔너리는 .item() 필요

print(f"시간 샘플: {len(time)}")
print(f"지층 수: {len(model['name'])}")

data.close()  # 사용 후 닫기

# 예제 2: with 구문 사용 (권장)
with np.load('marine_seismic_data.npz', allow_pickle=True) as data:
    time = data['time']
    clean = data['clean']
    model = data['model'].item()
    # 데이터 사용...

# 예제 3: 특정 키만 로드
data = np.load('marine_seismic_data.npz', allow_pickle=True)
if 'clean' in data:
    clean = data['clean']
    print("Clean 데이터 로드 성공")
else:
    print("Clean 데이터가 파일에 없습니다")
'''
    
    print(code_example)

def common_errors():
    """자주 발생하는 오류와 해결방법"""
    print("\n" + "="*70)
    print("자주 발생하는 오류와 해결방법")
    print("="*70)
    
    errors = [
        {
            "error": "FileNotFoundError: No such file or directory",
            "cause": "파일이 존재하지 않거나 경로가 잘못됨",
            "solution": [
                "1. 파일 경로 확인: os.path.exists('파일명')",
                "2. 작업 디렉토리 확인: os.getcwd()",
                "3. 시뮬레이션 실행하여 파일 생성"
            ]
        },
        {
            "error": "ValueError: Cannot load file containing pickled data",
            "cause": "allow_pickle=True 옵션이 없음",
            "solution": [
                "1. np.load(filename, allow_pickle=True) 사용",
                "2. 모델 딕셔너리 로드 시 .item() 호출"
            ]
        },
        {
            "error": "KeyError: 'model' or 'clean'",
            "cause": "파일에 해당 키가 없음",
            "solution": [
                "1. data.keys()로 사용 가능한 키 확인",
                "2. 올바른 파일인지 확인",
                "3. 파일 재생성"
            ]
        },
        {
            "error": "ImportError: No module named 'marine_seismic_simulation'",
            "cause": "모듈을 찾을 수 없음",
            "solution": [
                "1. 작업 디렉토리가 올바른지 확인",
                "2. marine_seismic_simulation.py 파일이 있는지 확인",
                "3. PYTHONPATH 설정 확인"
            ]
        },
        {
            "error": "AttributeError: 'dict' object has no attribute 'item'",
            "cause": "이미 딕셔너리인 객체에 .item() 호출",
            "solution": [
                "1. 객체 타입 확인: type(model)",
                "2. 이미 딕셔너리면 .item() 생략"
            ]
        }
    ]
    
    for i, err in enumerate(errors, 1):
        print(f"\n{i}. {err['error']}")
        print(f"   원인: {err['cause']}")
        print(f"   해결방법:")
        for sol in err['solution']:
            print(f"      {sol}")

def run_diagnostics():
    """진단 실행"""
    print("\n" + "="*70)
    print("파일 로드 진단 시작")
    print("="*70)
    
    # 1. 현재 디렉토리 확인
    print(f"\n[현재 디렉토리]")
    print(f"  {os.getcwd()}")
    
    # 2. 파일 목록
    print(f"\n[사용 가능한 .npz 파일]")
    npz_files = [f for f in os.listdir('.') if f.endswith('.npz')]
    if npz_files:
        for f in npz_files:
            size = os.path.getsize(f)
            print(f"  • {f} ({size:,} bytes)")
    else:
        print(f"  (없음)")
        print(f"  → 시뮬레이션을 먼저 실행하세요:")
        print(f"     python marine_seismic_simulation.py")
    
    # 3. 각 파일 로드 테스트
    print(f"\n[파일 로드 테스트]")
    for filename in npz_files:
        data = load_npz_safely(filename)
        if data is not None:
            # 모델이 있으면 로드 시도
            if 'model' in data:
                load_model_from_npz(data)
            data.close()

def main():
    """메인 함수"""
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "    파일 로드 문제 해결 가이드".center(68) + "║")
    print("║" + "    Troubleshooting Guide for File Loading".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # 진단 실행
    run_diagnostics()
    
    # 사용 예제
    example_usage()
    
    # 자주 발생하는 오류
    common_errors()
    
    print("\n" + "="*70)
    print("추가 도움이 필요하시면 구체적인 오류 메시지를 알려주세요!")
    print("="*70)

if __name__ == "__main__":
    main()
