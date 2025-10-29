# 🌊 Google Colab 실행 가이드

## ✅ 해결책: 독립 실행형 노트북 사용

"Failed to fetch" 오류가 발생하는 경우, **완전 독립 실행형 노트북**을 사용하세요!

---

## 🚀 방법 1: 독립 실행형 노트북 (추천!)

### 📌 특징
- ✅ 모든 코드가 노트북에 내장
- ✅ GitHub 접근 불필요
- ✅ 외부 파일 다운로드 없음
- ✅ 100% 작동 보장

### 🔗 링크
👉 **[Marine_Seismic_Standalone.ipynb 열기](https://colab.research.google.com/github/knocgp/seismic/blob/main/Marine_Seismic_Standalone.ipynb)**

### 📝 사용법
1. 위 링크 클릭
2. "런타임" > "모두 실행" 클릭
3. 결과 확인 및 다운로드

---

## 🔧 방법 2: 수동 실행 (대체 방법)

Colab 노트북이 열리지 않는 경우:

### 1️⃣ 새 노트북 생성
- Google Colab 접속: https://colab.research.google.com
- "새 노트북" 클릭

### 2️⃣ 첫 번째 셀에 코드 붙여넣기
```python
# GitHub에서 코드 다운로드 및 실행
!git clone https://github.com/knocgp/seismic.git
%cd seismic
!pip install -q numpy scipy matplotlib
!python marine_seismic_simulation.py
```

### 3️⃣ 실행
- Shift + Enter 또는 ▶ 버튼 클릭
- 시뮬레이션 자동 실행

### 4️⃣ 커스텀 파라미터 (선택사항)
두 번째 셀에 추가:
```python
from marine_seismic_simulation import MarineSeismicSimulator

# 파라미터 설정
sim = MarineSeismicSimulator(dt=0.001, nt=2000)
model = sim.create_synthetic_model(nlayers=6)
clean = sim.generate_synthetic_seismic(model, freq=40.0)
noisy = sim.add_marine_noise(clean, noise_level=0.10)

# 시각화
sim.plot_seismic_comparison({'Clean': clean, 'Noisy': noisy})
```

---

## 🆚 노트북 비교

| 특징 | Standalone | 기존 버전 |
|------|------------|-----------|
| **외부 의존성** | ❌ 없음 | ✅ GitHub 필요 |
| **실행 속도** | ⚡ 빠름 | 🐢 느림 |
| **오류 발생** | ❌ 없음 | ⚠️ "Failed to fetch" |
| **사용 편의성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **권장 여부** | ✅ 강력 추천 | ⚠️ 참고용 |

---

## ❓ 자주 묻는 질문

### Q1: "Failed to fetch" 오류가 발생합니다
**A:** Standalone 버전을 사용하세요:
- https://colab.research.google.com/github/knocgp/seismic/blob/main/Marine_Seismic_Standalone.ipynb

### Q2: 노트북 파일이 열리지 않습니다
**A:** 방법 2 (수동 실행)를 사용하세요:
1. 새 Colab 노트북 생성
2. 위의 코드 복사-붙여넣기
3. 실행

### Q3: GitHub 저장소에 접근할 수 없습니다
**A:** Standalone 노트북은 GitHub 접근이 필요 없습니다. 모든 코드가 내장되어 있습니다.

### Q4: 파라미터를 변경하고 싶습니다
**A:** Standalone 노트북의 "4. 커스텀 파라미터 시뮬레이션" 셀을 수정하세요:
```python
WAVELET_FREQ = 40.0         # 주파수 변경
MULTIPLE_STRENGTH = 0.7     # 멀티플 강도 변경
NOISE_LEVEL = 0.12          # 노이즈 레벨 변경
NUM_LAYERS = 7              # 지층 개수 변경
```

---

## 📚 추가 리소스

- **GitHub 저장소**: https://github.com/knocgp/seismic
- **README**: [전체 문서](https://github.com/knocgp/seismic/blob/main/README.md)
- **FAQ**: [자주 묻는 질문](https://github.com/knocgp/seismic/blob/main/FAQ.md)
- **로컬 실행**: [설치 및 실행 가이드](https://github.com/knocgp/seismic#설치-및-실행)

---

## 💡 팁

### 🎯 최적 파라미터
```python
# 일반적인 해상 탐사
WAVELET_FREQ = 30.0         # 30 Hz (표준)
MULTIPLE_STRENGTH = 0.5     # 중간 강도
NOISE_LEVEL = 0.08          # 약간의 노이즈

# 고해상도 탐사
WAVELET_FREQ = 50.0         # 고주파
MULTIPLE_STRENGTH = 0.3     # 약한 멀티플
NOISE_LEVEL = 0.05          # 적은 노이즈

# 도전적인 환경
WAVELET_FREQ = 20.0         # 저주파
MULTIPLE_STRENGTH = 0.8     # 강한 멀티플
NOISE_LEVEL = 0.15          # 높은 노이즈
```

### 📊 데이터 저장 및 로드
```python
# 데이터 저장
import numpy as np
np.savez('my_simulation.npz',
         time=simulator.time,
         clean=clean_seismic,
         noisy=noisy_seismic,
         model=model)

# 데이터 로드
data = np.load('my_simulation.npz', allow_pickle=True)
time = data['time']
clean = data['clean']
model = data['model'].item()
```

### 🎨 시각화 커스터마이징
```python
import matplotlib.pyplot as plt

# 커스텀 플롯
plt.figure(figsize=(15, 8))
plt.plot(simulator.time, clean_seismic, 'b-', label='Clean')
plt.plot(simulator.time, noisy_seismic, 'r-', alpha=0.5, label='Noisy')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title('Marine Seismic Survey Simulation', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

---

**Made with ❤️ for Marine Seismic Research**

**문의사항이 있으시면 GitHub 이슈를 등록해주세요!**
