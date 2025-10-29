# 해상 탄성파 탐사 시뮬레이션 (Marine Seismic Survey Simulation)

해상 탄성파 탐사의 합성 데이터 생성 및 시뮬레이션 프로그램입니다.

## 주요 기능

### 1. 합성 지반 모델 생성
- 해수층을 포함한 다층 지반 모델
- 각 층의 속도(Velocity), 밀도(Density), 두께(Thickness) 정의
- 실제 해저 지질구조를 반영한 물성 값

### 2. 합성 탄성파 데이터 생성
- **Reflectivity Method** 기반
- 반사 계수(Reflection Coefficient) 계산
- Ricker Wavelet 사용
- 컨볼루션을 통한 합성 탄성파 생성

### 3. 멀티플(Multiples) 시뮬레이션

#### 해면 멀티플 (Sea Surface Multiple)
- 해수면에서 반사되어 돌아오는 신호
- 1차, 2차 멀티플 포함
- 해면 반사 계수: -0.95 (공기-물 경계)

#### 내부 멀티플 (Internal Multiples)
- 지층 경계면 사이의 다중 반사
- 강한 반사면 간의 상호작용 시뮬레이션

### 4. 해상 탄성파 특유의 노이즈

#### 백색 잡음 (White Noise)
- 전자 장비에서 발생하는 랜덤 노이즈

#### 선박 노이즈 (Ship Noise)
- 저주파 대역 (2-8 Hz)
- 선박 엔진 및 프로펠러 소음

#### 해양 환경 노이즈 (Ocean Ambient Noise)
- 1-20 Hz 대역
- 해류, 파도 등 자연적 소음

#### 스웰 노이즈 (Swell Noise)
- 0.1-0.5 Hz 극저주파
- 해수면 파도에 의한 진폭 변조

#### 버스트 노이즈 (Burst Noise)
- 간헐적 충격 노이즈
- 해양 생물, 부유물 충돌 등

## 설치 및 실행

### 요구사항
```bash
pip install -r requirements.txt
```

필요한 패키지:
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

### 실행
```bash
python marine_seismic_simulation.py
```

## 출력 결과

### 생성되는 파일

1. **marine_seismic_model.png**
   - 속도 모델 및 밀도 모델 시각화
   - 각 지층의 깊이별 물성 분포

2. **marine_seismic_comparison.png**
   - Clean 탄성파 데이터
   - 멀티플이 추가된 데이터
   - 노이즈가 추가된 최종 데이터
   - 3가지 케이스 비교

3. **marine_seismic_spectrum.png**
   - 각 단계별 주파수 스펙트럼 분석
   - 0-100 Hz 범위

4. **marine_seismic_data.npz**
   - 모든 데이터를 NumPy 형식으로 저장
   - 포함 내용: time, clean, with_multiples, noisy, model

### 데이터 로드 예제
```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
data = np.load('marine_seismic_data.npz', allow_pickle=True)

time = data['time']
clean_seismic = data['clean']
with_multiples = data['with_multiples']
noisy_seismic = data['noisy']
model = data['model'].item()

# 플롯 예제
plt.figure(figsize=(10, 6))
plt.plot(time, clean_seismic, label='Clean')
plt.plot(time, noisy_seismic, label='Noisy', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Seismic Traces')
plt.grid(True)
plt.show()
```

## 클래스 및 메서드

### MarineSeismicSimulator 클래스

#### 초기화
```python
simulator = MarineSeismicSimulator(dt=0.001, nt=2000)
```
- `dt`: 샘플링 간격 (초), 기본값 0.001 (1 ms)
- `nt`: 시간 샘플 개수, 기본값 2000

#### 주요 메서드

##### 1. create_synthetic_model(nlayers=5)
합성 지반 모델 생성
- `nlayers`: 지층 개수 (해수층 포함)
- 반환: 지층 모델 딕셔너리

##### 2. generate_synthetic_seismic(model, freq=30.0)
합성 탄성파 데이터 생성
- `model`: 지층 모델
- `freq`: Wavelet 중심 주파수 (Hz)
- 반환: 합성 탄성파 데이터

##### 3. add_sea_surface_multiple(seismic, model, strength=0.5)
해면 멀티플 추가
- `seismic`: 원본 탄성파 데이터
- `model`: 지층 모델
- `strength`: 멀티플 강도 (0~1)
- 반환: 멀티플이 추가된 데이터

##### 4. add_internal_multiples(seismic, model, strength=0.3)
내부 멀티플 추가
- `seismic`: 원본 탄성파 데이터
- `model`: 지층 모델
- `strength`: 멀티플 강도
- 반환: 내부 멀티플이 추가된 데이터

##### 5. add_marine_noise(seismic, noise_level=0.05)
해상 노이즈 추가
- `seismic`: 원본 탄성파 데이터
- `noise_level`: 노이즈 레벨 (신호 대비)
- 반환: 노이즈가 추가된 데이터

## 사용 예제

### 기본 사용
```python
from marine_seismic_simulation import MarineSeismicSimulator

# 시뮬레이터 초기화
sim = MarineSeismicSimulator(dt=0.001, nt=2000)

# 지반 모델 생성
model = sim.create_synthetic_model(nlayers=6)

# 합성 탄성파 생성
clean = sim.generate_synthetic_seismic(model, freq=30.0)

# 멀티플 추가
with_multiples = sim.add_sea_surface_multiple(clean, model, strength=0.5)
with_multiples = sim.add_internal_multiples(with_multiples, model, strength=0.3)

# 노이즈 추가
noisy = sim.add_marine_noise(with_multiples, noise_level=0.08)

# 시각화
sim.plot_model(model)
sim.plot_seismic_comparison({
    'Clean': clean,
    'Noisy': noisy
})
```

### 커스텀 파라미터
```python
# 고주파 데이터 생성
high_freq = sim.generate_synthetic_seismic(model, freq=50.0)

# 강한 멀티플
strong_multiples = sim.add_sea_surface_multiple(clean, model, strength=0.8)

# 높은 노이즈 레벨
very_noisy = sim.add_marine_noise(clean, noise_level=0.15)
```

## 기술적 세부사항

### 반사 계수 계산
```
RC = (Z2 - Z1) / (Z2 + Z1)
```
여기서 Z는 음향 임피던스 (Acoustic Impedance):
```
Z = ρ × V
```
- ρ: 밀도 (kg/m³)
- V: 속도 (m/s)

### Ricker Wavelet
```
w(t) = (1 - 2a) × exp(-a)
a = (π × f × t)²
```
- f: 중심 주파수
- t: 시간

### 양방향 주시 (Two-Way Travel Time)
```
TWT = 2 × d / v
```
- d: 깊이 (m)
- v: 속도 (m/s)

## 참고 자료

### 해상 탄성파 탐사
- 에어건을 이용한 음파 송신
- 스트리머를 통한 반사파 수신
- 지하 구조 영상화

### 멀티플 제거 기법
- SRME (Surface Related Multiple Elimination)
- Radon Transform
- FX Deconvolution

### 노이즈 제거 기법
- F-K Filtering
- Median Filtering
- Adaptive Filtering

## 라이센스

MIT License

## 기여

이슈 및 풀 리퀘스트를 환영합니다.

## 문의

문제나 질문이 있으시면 이슈를 등록해주세요.
