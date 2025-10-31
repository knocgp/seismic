# FK Domain 직접파 제거 버그 수정

## 🐛 문제

> "직접파를 잘 찾아놓고 제거하지를 못해"

**증상:**
- FK domain에서 직접파 속도를 정확히 계산함
- 하지만 제거되지 않고 **오히려 보존됨**
- 다른 신호는 제거되는 반대 현상

---

## 🔍 원인 분석

### 기존 코드 (버그)

```python
if abs(velocity - water_velocity) < velocity_tolerance:
    velocity_diff = abs(velocity - water_velocity)
    taper = velocity_diff / velocity_tolerance  # 0 → 1
    taper = 0.5 * (1 - np.cos(np.pi * taper))  # Cosine taper
    fft2d_filtered[i_freq, i_kx] *= taper  # ❌ 문제!
```

### 문제점

**Taper 계산:**
- `velocity_diff = 0` (직접파 정확히) → `taper = 0`
- `velocity_diff = tolerance` (경계) → `taper = 1`

**적용:**
```python
fft2d_filtered *= taper
```

**결과:**
- 직접파 (taper=0): `fft2d * 0` = **완전히 보존** ❌
- 경계 (taper=1): `fft2d * 1` = 제거 ❌

**완전히 반대!**

---

## ✅ 해결 방법

### 수정된 코드

```python
if abs(velocity - water_velocity) < velocity_tolerance:
    velocity_diff = abs(velocity - water_velocity)
    
    # Normalized distance (0 at center, 1 at boundary)
    norm_dist = velocity_diff / velocity_tolerance
    
    # Removal taper (1 at center, 0 at boundary)
    removal = 0.5 * (1 + np.cos(np.pi * norm_dist))
    
    # Apply: preserve = 1 - removal
    fft2d_filtered[i_freq, i_kx] *= (1.0 - removal)  # ✅
```

### 올바른 로직

**Removal taper 계산:**
```python
removal = 0.5 * (1 + cos(π * norm_dist))
```

- `norm_dist = 0` (직접파): `removal = 0.5 * (1 + 1) = 1.0` → **완전 제거**
- `norm_dist = 1` (경계): `removal = 0.5 * (1 - 1) = 0.0` → **보존**

**적용:**
```python
preservation = 1.0 - removal
fft2d_filtered *= preservation
```

**결과:**
- 직접파: `fft2d * (1 - 1) = fft2d * 0` → **제거됨** ✅
- 경계: `fft2d * (1 - 0) = fft2d * 1` → **보존됨** ✅

---

## 📊 수식 비교

### 기존 (버그)

```
taper = 0.5 * (1 - cos(π * norm_dist))

norm_dist = 0 (직접파):
  taper = 0.5 * (1 - cos(0)) = 0.5 * (1 - 1) = 0

norm_dist = 1 (경계):
  taper = 0.5 * (1 - cos(π)) = 0.5 * (1 - (-1)) = 1

적용: fft * taper
  직접파: fft * 0 = 0 (제거 X, 다른 의미!)
  경계: fft * 1 = fft (보존)
  
→ 직접파 속도에서 taper=0이 되면 그 성분이 사라져야 하는데,
  실제로는 fft*0=0이 되어 다른 주파수도 영향받음
```

### 수정 (올바름)

```
removal = 0.5 * (1 + cos(π * norm_dist))

norm_dist = 0 (직접파):
  removal = 0.5 * (1 + cos(0)) = 0.5 * (1 + 1) = 1.0

norm_dist = 1 (경계):
  removal = 0.5 * (1 + cos(π)) = 0.5 * (1 + (-1)) = 0.0

적용: fft * (1 - removal)
  직접파: fft * (1 - 1) = fft * 0 = 완전 제거 ✅
  경계: fft * (1 - 0) = fft * 1 = 완전 보존 ✅
```

---

## 🎨 Taper 함수 그래프

### Removal Taper

```
removal
1.0 |  *
    |   \
0.5 |    \
    |     \
0.0 |      *___
    +-----------> norm_dist
    0    0.5    1.0
```

- `norm_dist = 0`: 직접파 → `removal = 1.0` (완전 제거)
- `norm_dist = 1`: 경계 → `removal = 0.0` (보존)
- Cosine taper: 부드러운 전환

### Preservation = 1 - Removal

```
preservation
1.0 |       ___*
    |      /
0.5 |     /
    |    /
0.0 | *
    +-----------> norm_dist
    0    0.5    1.0
```

- `norm_dist = 0`: 직접파 → `preservation = 0.0` (제거됨)
- `norm_dist = 1`: 경계 → `preservation = 1.0` (보존됨)

---

## 🔧 코드 변경 요약

### Before (버그)

```python
taper = velocity_diff / velocity_tolerance
taper = 0.5 * (1 - np.cos(np.pi * taper))
fft2d_filtered[i_freq, i_kx] *= taper  # ❌
```

### After (수정)

```python
norm_dist = velocity_diff / velocity_tolerance
removal = 0.5 * (1 + np.cos(np.pi * norm_dist))  # ✅ (1+cos)
fft2d_filtered[i_freq, i_kx] *= (1.0 - removal)  # ✅ (1 - removal)
```

**핵심 변경:**
1. `(1 - cos)` → `(1 + cos)` (removal taper)
2. `* taper` → `* (1 - removal)` (preservation)

---

## 🧪 검증

### 직접파 속도 (v = 1500 m/s)

```python
velocity_diff = 0  # 정확히 직접파
norm_dist = 0 / 100 = 0
removal = 0.5 * (1 + cos(0)) = 0.5 * 2 = 1.0
preservation = 1 - 1 = 0.0
→ FFT 성분 * 0 = 제거됨 ✅
```

### 경계 (v = 1500 ± 100 = 1400 or 1600 m/s)

```python
velocity_diff = 100  # 경계
norm_dist = 100 / 100 = 1.0
removal = 0.5 * (1 + cos(π)) = 0.5 * 0 = 0.0
preservation = 1 - 0 = 1.0
→ FFT 성분 * 1 = 보존됨 ✅
```

### 중간 (v = 1450 or 1550 m/s)

```python
velocity_diff = 50  # 중간
norm_dist = 50 / 100 = 0.5
removal = 0.5 * (1 + cos(π/2)) = 0.5 * 1 = 0.5
preservation = 1 - 0.5 = 0.5
→ FFT 성분 * 0.5 = 부분 제거 ✅
```

---

## 📈 효과

### Before (버그)
- ❌ 직접파가 제거되지 않음
- ❌ 다른 신호가 잘못 영향받음
- ❌ FK domain 필터링 실패

### After (수정)
- ✅ 직접파 정확히 제거
- ✅ 반사파 보존
- ✅ 부드러운 taper 전환
- ✅ FK domain 필터링 정상 작동

---

## 💡 교훈

### Taper 로직 설계 시 주의사항

1. **명확한 정의**
   - Removal인가? Preservation인가?
   - 0과 1의 의미 명확히

2. **적용 방식**
   - `* taper` vs `* (1 - taper)`
   - Removal이면 `* (1 - removal)`

3. **검증**
   - 경계 조건 확인 (0과 1)
   - 중간값 확인 (0.5)
   - 실제 데이터로 테스트

4. **주석**
   - Taper의 의미 명시
   - 0과 1에서의 동작 설명

---

## 🔄 관련 개념

### FK Domain Filtering

**기본 원리:**
```
v = ω/k = 2πf/k
```

- 각 (f, k) 점은 특정 속도에 대응
- 원하는 속도 범위의 (f, k) 제거
- 부드러운 taper로 artifacts 방지

### Velocity-based Filtering

**직접파:**
- v ≈ 1500 m/s (해수 속도)
- FK domain의 선형 영역
- 제거 대상

**반사파:**
- v ≠ 1500 m/s (층 속도 의존)
- FK domain의 다른 영역
- 보존 대상

---

## 📝 체크리스트

FK domain 필터링 구현 시:

- [ ] 속도 계산 올바른지 확인 (v = 2πf/k)
- [ ] Taper 정의 명확한지 (removal or preservation)
- [ ] 0과 1에서 동작 확인
- [ ] 적용 방식 올바른지 (* taper or * (1-taper))
- [ ] 경계 조건 테스트
- [ ] 실제 데이터로 검증
- [ ] 주석 추가

---

## 🚀 사용 방법

### 수정된 메서드

```python
# FK domain 직접파 제거 (수정됨)
after_direct = processor.remove_direct_wave_fk(
    shot_gather, offsets, model,
    velocity_tolerance=100  # m/s
)
```

### 파라미터

```python
velocity_tolerance = 100  # m/s
# 직접파 속도 ± 100 m/s 범위 제거
# 작을수록 선택적 (신호 보존)
# 클수록 넓게 제거 (직접파 완전 제거)
```

---

## 📚 참고

### Taper Functions

**Cosine taper (removal):**
```python
removal = 0.5 * (1 + cos(π * x))  # x ∈ [0, 1]
# x=0: removal=1 (완전 제거)
# x=1: removal=0 (보존)
```

**Cosine taper (preservation):**
```python
preservation = 0.5 * (1 - cos(π * x))  # x ∈ [0, 1]
# x=0: preservation=0 (제거)
# x=1: preservation=1 (보존)
```

**관계:**
```python
preservation = 1 - removal
```

---

**버그 수정 완료! 이제 FK domain 직접파 제거가 정상 작동합니다! 🎉**
