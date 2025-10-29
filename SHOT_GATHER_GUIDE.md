# 🎯 Shot Gather 워크플로우 가이드

## 📋 개요

완전 자동화된 Shot Gather 생성 및 노이즈 제거 워크플로우입니다.

---

## 🚀 빠른 시작

### 🌟 Google Colab에서 실행 (가장 쉬움!)

👉 **[Colab에서 바로 실행하기](https://colab.research.google.com/github/knocgp/seismic/blob/main/Shot_Gather_Workflow.ipynb)**

#### 실행 방법:
1. 위 링크 클릭
2. "런타임" > "모두 실행" 클릭
3. 자동으로 전체 워크플로우 실행
4. 결과 다운로드

---

## 📊 전체 워크플로우

### 1️⃣ 랜덤 합성 모델 생성
```
- 해수층 포함 완전 랜덤 지층 모델
- 지층 개수: 4-8개 (랜덤)
- 속도: 1500-5000 m/s
- 밀도: 1030-2800 kg/m³
- 두께: 150-600 m (층별 랜덤)
```

### 2️⃣ Shot Gather 생성
```
- 트레이스 개수: 48개 (기본값)
- 오프셋 범위: 100-2400 m
- NMO 효과 포함 (쌍곡선 이동시간)
- AVO 효과 포함 (각도 의존성)
- Geometric spreading 보정
```

### 3️⃣ 노이즈 추가
```
1. 백색 잡음 - 전자 장비 노이즈
2. Ground Roll - 저속 표면파 (5-15 Hz)
3. 스파이크 노이즈 - Bad traces
4. 저주파 트렌드 - 환경 노이즈
```

### 4️⃣ 노이즈 제거
```
1. 밴드패스 필터 (8-60 Hz)
   - 저주파 노이즈 제거
   - 고주파 노이즈 제거

2. F-K 필터
   - Ground Roll 제거
   - 속도 기반 필터링 (< 1500 m/s 억제)

3. Median 필터
   - 스파이크 노이즈 제거
   - Bad trace 보정
```

### 5️⃣ 비교 및 다운로드
```
생성되는 파일:
- shot_gather_clean.npz     (원본)
- shot_gather_noisy.npz     (노이즈 추가)
- shot_gather_denoised.npz  (노이즈 제거)
```

---

## 💻 로컬에서 실행

### 설치
```bash
git clone https://github.com/knocgp/seismic.git
cd seismic
pip install -r requirements.txt
```

### 실행
```bash
python shot_gather_workflow.py
```

---

## ⚙️ 커스텀 파라미터

### Python 스크립트에서
```python
from shot_gather_workflow import ShotGatherProcessor

# 프로세서 생성
processor = ShotGatherProcessor(dt=0.002, nt=1500)

# 모델 생성
model = processor.create_random_model(nlayers=7)

# Shot Gather 생성
clean, offsets = processor.generate_shot_gather(
    model, 
    n_traces=60,           # 트레이스 개수
    offset_min=50,         # 최소 오프셋 (m)
    offset_max=3000,       # 최대 오프셋 (m)
    freq=30.0              # Wavelet 주파수 (Hz)
)

# 노이즈 추가
noisy = processor.add_realistic_noise(clean, noise_level=0.15)

# 노이즈 제거
denoised = processor.denoise_combined(noisy)

# 시각화
processor.plot_comparison(clean, noisy, denoised, offsets)
```

### Colab 노트북에서
```python
# 커스텀 파라미터 설정
N_TRACES = 60           # 트레이스 개수
OFFSET_MIN = 50         # 최소 오프셋 (m)
OFFSET_MAX = 3000       # 최대 오프셋 (m)
WAVELET_FREQ = 30.0     # Wavelet 주파수 (Hz)
NOISE_LEVEL = 0.15      # 노이즈 레벨 (0-1)
N_LAYERS = 7            # 지층 개수

# 노트북의 "5. 커스텀 파라미터로 재실행" 셀에서 실행
```

---

## 📁 데이터 구조

### NPZ 파일 내용
```python
data = np.load('shot_gather_clean.npz', allow_pickle=True)

# 포함된 데이터:
shot_gather = data['shot_gather']  # Shape: (nt, n_traces)
offsets = data['offsets']           # Shape: (n_traces,)
time = data['time']                 # Shape: (nt,)
model = data['model'].item()        # Dictionary
```

### 모델 딕셔너리 구조
```python
model = {
    'velocity': [1500.0, 1800.0, ...],      # m/s
    'density': [1030.0, 2000.0, ...],       # kg/m³
    'thickness': [500.0, 300.0, ...],       # m
    'depth': [0.0, 500.0, ...],             # m
    'name': ['Water', 'Seabed', ...]        # 지층 이름
}
```

---

## 🔬 데이터 분석 예제

### 1. Shot Gather 로드 및 플롯
```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
data = np.load('shot_gather_clean.npz', allow_pickle=True)
shot = data['shot_gather']
offsets = data['offsets']
time = data['time']

# 플롯
plt.figure(figsize=(14, 10))
plt.imshow(shot, aspect='auto', cmap='seismic', 
           extent=[offsets[0], offsets[-1], time[-1], time[0]])
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.title('Shot Gather')
plt.colorbar(label='Amplitude')
plt.show()
```

### 2. 단일 트레이스 분석
```python
# 중간 오프셋 트레이스 선택
trace_idx = len(offsets) // 2

# 시간 도메인
plt.figure(figsize=(12, 6))
plt.plot(shot[:, trace_idx], time)
plt.xlabel('Amplitude')
plt.ylabel('Time (s)')
plt.title(f'Trace at Offset = {offsets[trace_idx]:.0f} m')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# 주파수 도메인
freq = np.fft.rfftfreq(len(time), time[1]-time[0])
spectrum = np.abs(np.fft.rfft(shot[:, trace_idx]))

plt.figure(figsize=(12, 6))
plt.plot(freq, spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude Spectrum')
plt.title('Frequency Spectrum')
plt.xlim([0, 100])
plt.grid(True)
plt.show()
```

### 3. SNR 계산
```python
# 데이터 로드
clean_data = np.load('shot_gather_clean.npz')
noisy_data = np.load('shot_gather_noisy.npz')
denoised_data = np.load('shot_gather_denoised.npz')

clean = clean_data['shot_gather']
noisy = noisy_data['shot_gather']
denoised = denoised_data['shot_gather']

# 노이즈 추출
noise_added = noisy - clean
noise_residual = denoised - clean

# SNR 계산
snr_before = 20 * np.log10(np.std(clean) / np.std(noise_added))
snr_after = 20 * np.log10(np.std(clean) / np.std(noise_residual))

print(f"SNR (노이즈 추가 후): {snr_before:.2f} dB")
print(f"SNR (노이즈 제거 후): {snr_after:.2f} dB")
print(f"SNR 개선: {snr_after - snr_before:.2f} dB")
```

---

## 🎓 이론적 배경

### Shot Gather
- **정의**: 하나의 소스 위치에서 여러 수신기가 기록한 데이터
- **오프셋**: 소스-수신기 거리
- **트레이스**: 각 수신기의 시계열 데이터

### NMO (Normal Moveout)
- **쌍곡선 이동시간**: t(x) = √(t₀² + x²/v²)
- t₀: 수직 반사 시간
- x: 오프셋
- v: 평균 속도

### AVO (Amplitude Variation with Offset)
- **각도 의존성**: A(θ) = A₀(1 - α·sin²θ)
- θ: 입사각
- α: AVO 계수

### Ground Roll
- **저속 표면파**: 300-800 m/s
- **주파수 범위**: 5-15 Hz
- **특징**: 높은 진폭, 긴 지속시간

---

## 🔧 고급 사용법

### 1. 개별 노이즈 제거 기법 적용

#### F-K 필터만 사용
```python
denoised_fk = processor.denoise_fk_filter(noisy, velocity_cutoff=1500)
```

#### 밴드패스 필터만 사용
```python
denoised_bp = processor.denoise_bandpass_filter(noisy, low_freq=8, high_freq=60)
```

#### Median 필터만 사용
```python
denoised_med = processor.denoise_median_filter(noisy, size=5)
```

### 2. 커스텀 노이즈 추가
```python
# 특정 노이즈만 추가
signal_power = np.std(clean)
nt, n_traces = clean.shape

# Ground Roll만 추가
ground_roll = np.zeros_like(clean)
for i in range(n_traces):
    freq = 10  # Hz
    phase_velocity = 500  # m/s
    offset = i * 50
    time_shift = offset / phase_velocity
    phase = 2 * np.pi * freq * (time - time_shift)
    ground_roll[:, i] = 0.5 * signal_power * np.sin(phase)

noisy_custom = clean + ground_roll
```

### 3. 배치 처리
```python
# 여러 Shot Gather 생성
n_shots = 10
all_shots = []

for i in range(n_shots):
    model = processor.create_random_model()
    shot, offsets = processor.generate_shot_gather(model)
    all_shots.append(shot)
    
# 저장
for i, shot in enumerate(all_shots):
    np.savez(f'shot_{i:03d}.npz', shot_gather=shot, offsets=offsets)
```

---

## 📊 결과 예시

### 전형적인 SNR 개선
```
SNR (노이즈 추가 후): 8-12 dB
SNR (노이즈 제거 후): 18-25 dB
SNR 개선: 10-15 dB
```

### 데이터 크기
```
트레이스 개수: 48
시간 샘플: 1500
파일 크기: ~600 KB (clean), ~700 KB (noisy), ~650 KB (denoised)
```

---

## ❓ FAQ

### Q1: Shot Gather가 너무 노이지합니다
**A:** `noise_level` 파라미터를 낮추세요:
```python
noisy = processor.add_realistic_noise(clean, noise_level=0.08)  # 기본 0.12
```

### Q2: 노이즈 제거가 너무 과하게 됩니다
**A:** 개별 필터를 조정하거나 순차적으로 적용하세요:
```python
# 부드러운 노이즈 제거
result = processor.denoise_bandpass_filter(noisy, 5, 80)  # 넓은 대역
result = processor.denoise_fk_filter(result, 1000)  # 낮은 컷오프
```

### Q3: Ground Roll이 제대로 제거되지 않습니다
**A:** F-K 필터의 속도 컷오프를 조정하세요:
```python
denoised = processor.denoise_fk_filter(noisy, velocity_cutoff=2000)  # 더 높은 값
```

### Q4: 더 많은 트레이스가 필요합니다
**A:** `n_traces` 파라미터를 증가시키세요:
```python
shot, offsets = processor.generate_shot_gather(model, n_traces=96)
```

### Q5: 다른 주파수 대역을 사용하고 싶습니다
**A:** Wavelet 주파수를 변경하세요:
```python
# 저주파 (심부 탐사)
shot = processor.generate_shot_gather(model, freq=15.0)

# 고주파 (천부 탐사)
shot = processor.generate_shot_gather(model, freq=50.0)
```

---

## 🔗 관련 링크

- **GitHub 저장소**: https://github.com/knocgp/seismic
- **메인 README**: [README.md](https://github.com/knocgp/seismic/blob/main/README.md)
- **Colab 가이드**: [COLAB_GUIDE_KR.md](https://github.com/knocgp/seismic/blob/main/COLAB_GUIDE_KR.md)
- **FAQ**: [FAQ.md](https://github.com/knocgp/seismic/blob/main/FAQ.md)

---

## 📝 참고 문헌

### 노이즈 제거 기법
- F-K Filtering: Yilmaz, O. (2001). Seismic Data Analysis
- Median Filtering: Bednar, J. B. (1983). Applications of median filtering to deconvolution
- Bandpass Filtering: Sheriff, R. E., & Geldart, L. P. (1995). Exploration Seismology

### Shot Gather 처리
- NMO Correction: Taner, M. T., & Koehler, F. (1969)
- AVO Analysis: Rutherford, S. R., & Williams, R. H. (1989)

---

**Made with ❤️ for Seismic Data Processing**

**문의사항이 있으시면 GitHub 이슈를 등록해주세요!**
