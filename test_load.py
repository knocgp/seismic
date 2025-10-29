#!/usr/bin/env python3
"""
파일 로드 테스트 스크립트
"""

import numpy as np
import sys

print("="*70)
print("파일 로드 테스트")
print("="*70)

# 1. 모듈 import 테스트
print("\n[1] 모듈 import 테스트...")
try:
    from marine_seismic_simulation import MarineSeismicSimulator
    print("✓ marine_seismic_simulation 모듈 import 성공")
except Exception as e:
    print(f"✗ 모듈 import 실패: {e}")
    sys.exit(1)

# 2. 시뮬레이터 생성 테스트
print("\n[2] 시뮬레이터 생성 테스트...")
try:
    sim = MarineSeismicSimulator(dt=0.001, nt=100)
    print("✓ 시뮬레이터 생성 성공")
except Exception as e:
    print(f"✗ 시뮬레이터 생성 실패: {e}")
    sys.exit(1)

# 3. NPZ 파일 로드 테스트
print("\n[3] NPZ 파일 로드 테스트...")
npz_files = [
    'marine_seismic_data.npz',
    'cmp_gather_data.npz',
    'quick_example_data.npz'
]

for npz_file in npz_files:
    try:
        data = np.load(npz_file, allow_pickle=True)
        keys = list(data.keys())
        print(f"✓ {npz_file} 로드 성공 - 키: {keys}")
        data.close()
    except FileNotFoundError:
        print(f"⚠ {npz_file} 파일 없음 (정상 - 아직 생성 안됨)")
    except Exception as e:
        print(f"✗ {npz_file} 로드 실패: {type(e).__name__}: {e}")

# 4. 간단한 시뮬레이션 테스트
print("\n[4] 간단한 시뮬레이션 테스트...")
try:
    model = sim.create_synthetic_model(nlayers=3)
    print(f"✓ 모델 생성 성공 - {len(model['name'])}개 층")
    
    seismic = sim.generate_synthetic_seismic(model, freq=30.0)
    print(f"✓ 탄성파 데이터 생성 성공 - {len(seismic)} 샘플")
    
    print(f"  데이터 형태: {seismic.shape}")
    print(f"  데이터 타입: {seismic.dtype}")
    print(f"  최대값: {np.max(seismic):.6f}")
    print(f"  최소값: {np.min(seismic):.6f}")
    
except Exception as e:
    print(f"✗ 시뮬레이션 실패: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 데이터 저장 및 로드 테스트
print("\n[5] 데이터 저장 및 로드 테스트...")
try:
    # 저장
    test_file = 'test_data.npz'
    np.savez(test_file,
             time=sim.time,
             seismic=seismic,
             model=model)
    print(f"✓ 테스트 데이터 저장 성공: {test_file}")
    
    # 로드
    loaded = np.load(test_file, allow_pickle=True)
    print(f"✓ 테스트 데이터 로드 성공")
    print(f"  키: {list(loaded.keys())}")
    print(f"  time shape: {loaded['time'].shape}")
    print(f"  seismic shape: {loaded['seismic'].shape}")
    
    loaded_model = loaded['model'].item()
    print(f"  model 타입: {type(loaded_model)}")
    print(f"  model 키: {list(loaded_model.keys())}")
    
    loaded.close()
    
except Exception as e:
    print(f"✗ 저장/로드 실패: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ 모든 테스트 완료!")
print("="*70)
