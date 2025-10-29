# 자주 묻는 질문 (FAQ)

## 파일 로드 관련

### Q1: "FileNotFoundError" 오류가 발생합니다

**증상:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'marine_seismic_data.npz'
```

**원인:** 데이터 파일이 아직 생성되지 않았습니다.

**해결방법:**
```bash
# 먼저 시뮬레이션을 실행하여 데이터를 생성하세요
python marine_seismic_simulation.py
```

---

### Q2: "ValueError: Cannot load file containing pickled data" 오류

**증상:**
```
ValueError: Cannot load file containing pickled data when allow_pickle=False
```

**원인:** NumPy 보안 설정으로 인해 pickle 데이터 로드가 거부됩니다.

**해결방법:**
```python
# allow_pickle=True 옵션 추가
data = np.load('marine_seismic_data.npz', allow_pickle=True)
```

---

### Q3: 모델 딕셔너리 로드 시 오류

**증상:**
```python
model = data['model']  # 이것은 0-dimensional array
print(model['velocity'])  # TypeError 발생
```

**원인:** NumPy 배열로 저장된 딕셔너리를 직접 사용하려고 함

**해결방법:**
```python
# .item() 메서드로 딕셔너리 추출
model = data['model'].item()
print(model['velocity'])  # 정상 작동
```

---

### Q4: "ModuleNotFoundError: No module named 'marine_seismic_simulation'" 오류

**증상:**
```
ModuleNotFoundError: No module named 'marine_seismic_simulation'
```

**원인:** 모듈 파일이 없거나 잘못된 디렉토리에서 실행

**해결방법:**
```bash
# 1. 올바른 디렉토리로 이동
cd /home/user/webapp

# 2. 파일 존재 확인
ls marine_seismic_simulation.py

# 3. Python 경로 확인
python -c "import sys; print('\n'.join(sys.path))"
```

---

## 실행 관련

### Q5: 그림이 표시되지 않습니다

**원인:** 백엔드 설정 또는 디스플레이 환경 문제

**해결방법:**
```python
# 코드 상단에 추가
import matplotlib
matplotlib.use('Agg')  # 파일로만 저장
import matplotlib.pyplot as plt
```

---

### Q6: 메모리 부족 오류

**증상:**
```
MemoryError: Unable to allocate array
```

**해결방법:**
```python
# 샘플 수 줄이기
sim = MarineSeismicSimulator(dt=0.002, nt=1000)  # 기본값의 절반
```

---

## 올바른 사용 예제

### 기본 데이터 로드
```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 (with 구문 권장)
with np.load('marine_seismic_data.npz', allow_pickle=True) as data:
    time = data['time']
    clean = data['clean']
    noisy = data['noisy']
    model = data['model'].item()  # 딕셔너리는 .item() 필요

# 2. 데이터 사용
print(f"시간 범위: 0 ~ {time[-1]:.2f} s")
print(f"지층 수: {len(model['name'])}")

# 3. 플롯
plt.figure(figsize=(10, 6))
plt.plot(time, clean, label='Clean')
plt.plot(time, noisy, label='Noisy', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.savefig('my_plot.png')
print("그림 저장: my_plot.png")
```

### 커스텀 시뮬레이션
```python
from marine_seismic_simulation import MarineSeismicSimulator

# 1. 시뮬레이터 생성
sim = MarineSeismicSimulator(dt=0.001, nt=2000)

# 2. 모델 생성
model = sim.create_synthetic_model(nlayers=5)

# 3. 데이터 생성
clean = sim.generate_synthetic_seismic(model, freq=30.0)

# 4. 멀티플 추가 (선택)
with_mult = sim.add_sea_surface_multiple(clean, model, strength=0.5)

# 5. 노이즈 추가 (선택)
noisy = sim.add_marine_noise(with_mult, noise_level=0.08)

# 6. 시각화
sim.plot_seismic_comparison({
    'Clean': clean,
    'Noisy': noisy
})
```

---

## 성능 최적화

### 빠른 시뮬레이션
```python
# 샘플 수를 줄여서 빠르게 테스트
sim = MarineSeismicSimulator(dt=0.002, nt=1000)
model = sim.create_synthetic_model(nlayers=4)
```

### 메모리 절약
```python
# 데이터 사용 후 즉시 닫기
data = np.load('file.npz', allow_pickle=True)
needed_data = data['clean'].copy()  # 필요한 것만 복사
data.close()  # 메모리 해제
```

---

## 추가 도움

더 자세한 정보는 다음을 참고하세요:
- README.md - 전체 문서
- demo_custom_simulation.py - 다양한 사용 예제
- quick_example.py - 빠른 시작 가이드

**문제가 계속되면:**
1. Python 버전 확인: `python --version` (3.7+ 필요)
2. 패키지 재설치: `pip install -r requirements.txt --upgrade`
3. 데이터 파일 재생성: `python marine_seismic_simulation.py`
