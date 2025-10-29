#!/usr/bin/env python3
"""
Shot Gather 생성 및 노이즈 제거 워크플로우
Shot Gather Generation and Denoising Workflow

전체 워크플로우:
1. 랜덤 합성 모델 생성
2. Shot Gather 생성 (다중 트레이스)
3. 노이즈 추가
4. 노이즈 제거 (여러 기법)
5. 비교 및 다운로드
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ShotGatherProcessor:
    """Shot Gather 생성 및 처리 클래스"""
    
    def __init__(self, dt: float = 0.002, nt: int = 1500):
        """
        초기화
        
        Parameters:
        -----------
        dt : float
            샘플링 간격 (초), 기본값 2 ms
        nt : int
            시간 샘플 개수
        """
        self.dt = dt
        self.nt = nt
        self.time = np.arange(nt) * dt
        
    def create_random_model(self, nlayers: int = None) -> Dict:
        """
        완전 랜덤 합성 지반 모델 생성
        
        Parameters:
        -----------
        nlayers : int
            지층 개수 (None이면 3-8 사이 랜덤)
            
        Returns:
        --------
        model : dict
            velocity : 속도 (m/s)
            density : 밀도 (kg/m³)
            thickness : 두께 (m)
            depth : 깊이 (m)
            name : 지층 이름
        """
        if nlayers is None:
            nlayers = np.random.randint(4, 9)  # 4-8개 층
        
        model = {
            'velocity': [],
            'density': [],
            'thickness': [],
            'depth': [],
            'name': []
        }
        
        # 1. 해수층 (항상 포함)
        water_depth = np.random.uniform(300, 800)  # 300-800m
        model['velocity'].append(1500.0)
        model['density'].append(1030.0)
        model['thickness'].append(water_depth)
        model['depth'].append(0.0)
        model['name'].append('Water')
        
        # 2. 해저면 (연약 퇴적층)
        seabed_vp = np.random.uniform(1600, 2000)
        seabed_rho = np.random.uniform(1900, 2100)
        seabed_thick = np.random.uniform(200, 400)
        
        model['velocity'].append(seabed_vp)
        model['density'].append(seabed_rho)
        model['thickness'].append(seabed_thick)
        model['depth'].append(water_depth)
        model['name'].append('Seabed')
        
        # 3-N. 지하 지층들 (완전 랜덤)
        current_depth = water_depth + seabed_thick
        
        for i in range(nlayers - 2):
            # 속도: 이전 층보다 증가하되 랜덤 변동
            if i == 0:
                base_vp = seabed_vp + np.random.uniform(200, 500)
            else:
                base_vp = model['velocity'][-1] + np.random.uniform(100, 600)
            
            vp = base_vp + np.random.normal(0, 100)
            vp = np.clip(vp, 2000, 5000)  # 2000-5000 m/s
            
            # 밀도: 속도와 약한 상관관계
            rho = 2000 + (vp - 2000) * 0.2 + np.random.normal(0, 50)
            rho = np.clip(rho, 2000, 2800)
            
            # 두께: 완전 랜덤
            thickness = np.random.uniform(150, 600)
            
            model['velocity'].append(vp)
            model['density'].append(rho)
            model['thickness'].append(thickness)
            model['depth'].append(current_depth)
            model['name'].append(f'Layer {i+3}')
            
            current_depth += thickness
        
        return model
    
    def calculate_reflection_coefficients(self, model: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """반사 계수 및 시간 계산"""
        velocities = np.array(model['velocity'])
        densities = np.array(model['density'])
        thicknesses = np.array(model['thickness'])
        
        # 음향 임피던스
        impedance = velocities * densities
        
        # 반사 계수
        rc = np.zeros(len(velocities) - 1)
        for i in range(len(velocities) - 1):
            rc[i] = (impedance[i+1] - impedance[i]) / (impedance[i+1] + impedance[i])
        
        # 반사 시간 (양방향 주시)
        times = np.zeros(len(velocities) - 1)
        cumulative_time = 0
        for i in range(len(velocities) - 1):
            travel_time = thicknesses[i] / velocities[i]
            cumulative_time += travel_time
            times[i] = cumulative_time * 2
        
        return rc, times
    
    def ricker_wavelet(self, freq: float = 25.0) -> np.ndarray:
        """Ricker 파형 생성"""
        duration = 0.2
        t = np.arange(-duration/2, duration/2, self.dt)
        a = (np.pi * freq * t) ** 2
        wavelet = (1 - 2*a) * np.exp(-a)
        wavelet = wavelet / np.max(np.abs(wavelet))
        return wavelet
    
    def generate_shot_gather(self, model: Dict, n_traces: int = 48, 
                           offset_min: float = 100, offset_max: float = 2400,
                           freq: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shot Gather 생성
        
        Parameters:
        -----------
        model : dict
            지층 모델
        n_traces : int
            트레이스 개수
        offset_min : float
            최소 오프셋 (m)
        offset_max : float
            최대 오프셋 (m)
        freq : float
            Wavelet 주파수
            
        Returns:
        --------
        shot_gather : np.ndarray
            Shape (nt, n_traces)
        offsets : np.ndarray
            오프셋 배열
        """
        # 오프셋 배열
        offsets = np.linspace(offset_min, offset_max, n_traces)
        
        # Shot gather 초기화
        shot_gather = np.zeros((self.nt, n_traces))
        
        # Wavelet 생성
        wavelet = self.ricker_wavelet(freq)
        
        # 반사 계수 및 시간
        rc, zero_offset_times = self.calculate_reflection_coefficients(model)
        
        # 각 트레이스 생성
        for i_trace, offset in enumerate(offsets):
            # Reflectivity series
            reflectivity = np.zeros(self.nt)
            
            # 각 반사면에 대해
            for j, (rc_val, t0) in enumerate(zip(rc, zero_offset_times)):
                # 평균 속도 (간단화)
                depths = np.array(model['depth'])
                velocities = np.array(model['velocity'])
                
                if j < len(depths) - 1:
                    avg_depth = depths[j+1]
                    avg_velocity = np.mean(velocities[:j+2])
                    
                    # NMO 보정된 시간 (쌍곡선)
                    t_nmo = np.sqrt(t0**2 + (offset / avg_velocity)**2)
                    
                    # AVO 효과 (간단 버전)
                    angle = np.arctan(offset / avg_depth)
                    avo_factor = 1 - 0.3 * np.sin(angle)**2
                    
                    # Reflectivity에 추가
                    idx = int(t_nmo / self.dt)
                    if idx < self.nt:
                        reflectivity[idx] += rc_val * avo_factor
            
            # 컨볼루션
            trace = signal.convolve(reflectivity, wavelet, mode='same')
            
            # Geometric spreading 보정
            spreading = 1 / (1 + offset / 1000)
            trace *= spreading
            
            shot_gather[:, i_trace] = trace
        
        return shot_gather, offsets
    
    def add_realistic_noise(self, shot_gather: np.ndarray, 
                           noise_level: float = 0.10) -> np.ndarray:
        """
        실제적인 노이즈 추가
        
        Parameters:
        -----------
        shot_gather : np.ndarray
            원본 Shot Gather
        noise_level : float
            노이즈 레벨 (0-1)
            
        Returns:
        --------
        noisy_gather : np.ndarray
            노이즈가 추가된 Shot Gather
        """
        result = shot_gather.copy()
        signal_power = np.std(shot_gather)
        nt, n_traces = shot_gather.shape
        
        # 1. 백색 잡음
        white_noise = np.random.normal(0, noise_level * signal_power * 0.3, 
                                      (nt, n_traces))
        result += white_noise
        
        # 2. 수평 연속 노이즈 (Ground Roll 시뮬레이션)
        for i in range(5):
            freq = np.random.uniform(5, 15)  # 5-15 Hz
            phase_velocity = np.random.uniform(300, 800)  # m/s
            amplitude = noise_level * signal_power * np.random.uniform(0.5, 1.5)
            
            for j in range(n_traces):
                offset = j * 50  # 가정된 트레이스 간격
                time_shift = offset / phase_velocity
                phase = 2 * np.pi * freq * (self.time - time_shift)
                ground_roll = amplitude * np.sin(phase + np.random.uniform(0, 2*np.pi))
                
                # 시간에 따라 감쇠
                decay = np.exp(-self.time / 0.5)
                result[:, j] += ground_roll * decay
        
        # 3. 스파이크 노이즈 (Bad traces)
        n_spikes = np.random.randint(1, 4)
        for _ in range(n_spikes):
            spike_trace = np.random.randint(0, n_traces)
            spike_time = np.random.randint(0, nt)
            spike_duration = np.random.randint(20, 100)
            
            if spike_time + spike_duration < nt:
                spike = noise_level * signal_power * 5.0 * np.random.randn(spike_duration)
                result[spike_time:spike_time+spike_duration, spike_trace] += spike
        
        # 4. 저주파 트렌드
        for j in range(n_traces):
            trend_freq = np.random.uniform(0.5, 2.0)
            trend = noise_level * signal_power * 0.4 * np.sin(2 * np.pi * trend_freq * self.time)
            result[:, j] += trend
        
        return result
    
    def denoise_fk_filter(self, shot_gather: np.ndarray, 
                         velocity_cutoff: float = 1500) -> np.ndarray:
        """
        F-K 필터링으로 노이즈 제거
        
        Parameters:
        -----------
        shot_gather : np.ndarray
            노이즈가 있는 Shot Gather
        velocity_cutoff : float
            속도 컷오프 (m/s) - 이보다 느린 속도 제거
            
        Returns:
        --------
        denoised : np.ndarray
            노이즈 제거된 Shot Gather
        """
        nt, n_traces = shot_gather.shape
        
        # 2D FFT
        fk_spectrum = np.fft.fft2(shot_gather)
        fk_spectrum_shifted = np.fft.fftshift(fk_spectrum)
        
        # 주파수 및 파수 축
        freq = np.fft.fftshift(np.fft.fftfreq(nt, self.dt))
        k = np.fft.fftshift(np.fft.fftfreq(n_traces, 50))  # 50m 트레이스 간격 가정
        
        # F-K 도메인 필터 생성
        fk_filter = np.ones_like(fk_spectrum_shifted)
        
        for i, f in enumerate(freq):
            for j, kval in enumerate(k):
                if f != 0 and kval != 0:
                    apparent_velocity = abs(f / kval)
                    if apparent_velocity < velocity_cutoff:
                        fk_filter[i, j] = 0.1  # 저속 성분 억제
        
        # 필터 적용
        fk_filtered = fk_spectrum_shifted * fk_filter
        fk_filtered_unshifted = np.fft.ifftshift(fk_filtered)
        
        # 역 FFT
        denoised = np.real(np.fft.ifft2(fk_filtered_unshifted))
        
        return denoised
    
    def denoise_median_filter(self, shot_gather: np.ndarray, 
                             size: int = 5) -> np.ndarray:
        """
        Median 필터로 스파이크 노이즈 제거
        
        Parameters:
        -----------
        shot_gather : np.ndarray
            노이즈가 있는 Shot Gather
        size : int
            필터 크기
            
        Returns:
        --------
        denoised : np.ndarray
            노이즈 제거된 Shot Gather
        """
        denoised = median_filter(shot_gather, size=(size, 1))
        return denoised
    
    def denoise_bandpass_filter(self, shot_gather: np.ndarray,
                               low_freq: float = 8.0,
                               high_freq: float = 60.0) -> np.ndarray:
        """
        밴드패스 필터로 주파수 노이즈 제거
        
        Parameters:
        -----------
        shot_gather : np.ndarray
            노이즈가 있는 Shot Gather
        low_freq : float
            저주파 컷오프 (Hz)
        high_freq : float
            고주파 컷오프 (Hz)
            
        Returns:
        --------
        denoised : np.ndarray
            노이즈 제거된 Shot Gather
        """
        nt, n_traces = shot_gather.shape
        denoised = np.zeros_like(shot_gather)
        
        # Butterworth 밴드패스 필터
        nyquist = 1 / (2 * self.dt)
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # 각 트레이스에 적용
        for i in range(n_traces):
            denoised[:, i] = signal.filtfilt(b, a, shot_gather[:, i])
        
        return denoised
    
    def denoise_combined(self, shot_gather: np.ndarray) -> np.ndarray:
        """
        여러 기법 조합한 노이즈 제거
        
        Parameters:
        -----------
        shot_gather : np.ndarray
            노이즈가 있는 Shot Gather
            
        Returns:
        --------
        denoised : np.ndarray
            노이즈 제거된 Shot Gather
        """
        # 1. 밴드패스 필터 (주파수 노이즈)
        result = self.denoise_bandpass_filter(shot_gather, 8.0, 60.0)
        
        # 2. F-K 필터 (Ground Roll)
        result = self.denoise_fk_filter(result, 1500)
        
        # 3. Median 필터 (스파이크)
        result = self.denoise_median_filter(result, 5)
        
        return result
    
    def plot_shot_gather(self, shot_gather: np.ndarray, offsets: np.ndarray,
                        title: str = "Shot Gather", clip_percentile: float = 99):
        """Shot Gather 시각화"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Clip 설정
        vmax = np.percentile(np.abs(shot_gather), clip_percentile)
        
        # Wiggle trace plot
        for i, offset in enumerate(offsets):
            trace = shot_gather[:, i]
            trace_scaled = trace / vmax * 30  # 스케일링
            
            # Wiggle
            ax.plot(offset + trace_scaled, self.time, 'k-', linewidth=0.3)
            
            # Positive fill
            ax.fill_betweenx(self.time, offset, offset + trace_scaled,
                            where=(trace_scaled > 0), color='black', alpha=0.6)
        
        ax.set_xlabel('Offset (m)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Time (s)', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([offsets[0] - 100, offsets[-1] + 100])
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, original: np.ndarray, noisy: np.ndarray,
                       denoised: np.ndarray, offsets: np.ndarray):
        """원본, 노이즈, 노이즈 제거 비교"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        
        titles = ['Original (Clean)', 'With Noise', 'Denoised']
        data_list = [original, noisy, denoised]
        
        # Clip 설정
        vmax = np.percentile(np.abs(original), 99)
        
        for ax, data, title in zip(axes, data_list, titles):
            # Wiggle plot
            for i, offset in enumerate(offsets):
                trace = data[:, i]
                trace_scaled = trace / vmax * 30
                
                ax.plot(offset + trace_scaled, self.time, 'k-', linewidth=0.3)
                ax.fill_betweenx(self.time, offset, offset + trace_scaled,
                                where=(trace_scaled > 0), color='black', alpha=0.6)
            
            ax.set_xlabel('Offset (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim([offsets[0] - 100, offsets[-1] + 100])
        
        plt.tight_layout()
        plt.show()
    
    def plot_model(self, model: Dict):
        """지층 모델 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        depths = model['depth']
        velocities = model['velocity']
        densities = model['density']
        
        for i in range(len(depths)):
            depth_top = depths[i]
            depth_bottom = depths[i] + model['thickness'][i]
            
            # 속도 프로파일
            ax1.fill_between([velocities[i]-100, velocities[i]+100],
                            depth_top, depth_bottom,
                            alpha=0.4, label=model['name'][i] if i < 5 else None)
            ax1.plot([velocities[i], velocities[i]], [depth_top, depth_bottom],
                    'b-', linewidth=2.5)
            
            # 밀도 프로파일
            ax2.fill_between([densities[i]-50, densities[i]+50],
                            depth_top, depth_bottom,
                            alpha=0.4)
            ax2.plot([densities[i], densities[i]], [depth_top, depth_bottom],
                    'r-', linewidth=2.5)
        
        ax1.set_xlabel('Velocity (m/s)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Depth (m)', fontsize=13, fontweight='bold')
        ax1.set_title('Velocity Model', fontsize=15, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.4)
        ax1.legend(fontsize=10)
        
        ax2.set_xlabel('Density (kg/m³)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Depth (m)', fontsize=13, fontweight='bold')
        ax2.set_title('Density Model', fontsize=15, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.4)
        
        plt.tight_layout()
        plt.show()


def main():
    """메인 워크플로우"""
    print("="*80)
    print("Shot Gather 생성 및 노이즈 제거 워크플로우")
    print("Shot Gather Generation and Denoising Workflow")
    print("="*80)
    print()
    
    # 프로세서 초기화
    processor = ShotGatherProcessor(dt=0.002, nt=1500)
    print("✓ 프로세서 초기화 완료")
    
    # 1. 랜덤 합성 모델 생성
    print("\n[1단계] 랜덤 합성 지반 모델 생성...")
    model = processor.create_random_model(nlayers=6)
    
    print("\n📊 생성된 지층 정보:")
    print(f"{'Layer':<15} {'Depth (m)':<12} {'Thickness (m)':<15} "
          f"{'Velocity (m/s)':<15} {'Density (kg/m³)'}")
    print("-"*75)
    for i in range(len(model['name'])):
        print(f"{model['name'][i]:<15} {model['depth'][i]:<12.1f} "
              f"{model['thickness'][i]:<15.1f} {model['velocity'][i]:<15.1f} "
              f"{model['density'][i]:<15.1f}")
    
    # 모델 시각화
    print("\n📈 지층 모델 시각화...")
    processor.plot_model(model)
    
    # 2. Shot Gather 생성
    print("\n[2단계] Shot Gather 생성...")
    n_traces = 48
    clean_shot, offsets = processor.generate_shot_gather(
        model, n_traces=n_traces, 
        offset_min=100, offset_max=2400,
        freq=25.0
    )
    print(f"✓ {n_traces}개 트레이스 Shot Gather 생성 완료")
    print(f"  오프셋 범위: {offsets[0]:.0f} - {offsets[-1]:.0f} m")
    
    # Clean Shot 시각화
    print("\n📊 원본 Shot Gather 시각화...")
    processor.plot_shot_gather(clean_shot, offsets, "Original Shot Gather (Clean)")
    
    # 3. 노이즈 추가
    print("\n[3단계] 노이즈 추가...")
    noisy_shot = processor.add_realistic_noise(clean_shot, noise_level=0.12)
    print("✓ 백색 잡음 추가")
    print("✓ Ground Roll 추가")
    print("✓ 스파이크 노이즈 추가")
    print("✓ 저주파 트렌드 추가")
    
    # Noisy Shot 시각화
    print("\n📊 노이즈가 추가된 Shot Gather 시각화...")
    processor.plot_shot_gather(noisy_shot, offsets, "Shot Gather with Noise")
    
    # 4. 노이즈 제거
    print("\n[4단계] 노이즈 제거...")
    print("  - 밴드패스 필터 (8-60 Hz)")
    print("  - F-K 필터 (Ground Roll 제거)")
    print("  - Median 필터 (스파이크 제거)")
    
    denoised_shot = processor.denoise_combined(noisy_shot)
    print("✓ 노이즈 제거 완료")
    
    # Denoised Shot 시각화
    print("\n📊 노이즈 제거된 Shot Gather 시각화...")
    processor.plot_shot_gather(denoised_shot, offsets, "Denoised Shot Gather")
    
    # 5. 전체 비교
    print("\n[5단계] 전체 비교...")
    processor.plot_comparison(clean_shot, noisy_shot, denoised_shot, offsets)
    
    # 6. 데이터 저장
    print("\n[6단계] 데이터 저장...")
    np.savez('shot_gather_clean.npz',
             shot_gather=clean_shot,
             offsets=offsets,
             time=processor.time,
             model=model)
    print("✓ 원본 Shot Gather 저장: shot_gather_clean.npz")
    
    np.savez('shot_gather_noisy.npz',
             shot_gather=noisy_shot,
             offsets=offsets,
             time=processor.time,
             model=model)
    print("✓ 노이즈 Shot Gather 저장: shot_gather_noisy.npz")
    
    np.savez('shot_gather_denoised.npz',
             shot_gather=denoised_shot,
             offsets=offsets,
             time=processor.time,
             model=model)
    print("✓ 노이즈 제거 Shot Gather 저장: shot_gather_denoised.npz")
    
    # 7. 통계 정보
    print("\n" + "="*80)
    print("📊 데이터 통계")
    print("="*80)
    print(f"트레이스 개수: {n_traces}")
    print(f"시간 샘플: {processor.nt}")
    print(f"샘플링 간격: {processor.dt*1000:.1f} ms")
    print(f"총 시간: {processor.time[-1]:.2f} s")
    print(f"오프셋 범위: {offsets[0]:.0f} - {offsets[-1]:.0f} m")
    print(f"\nClean RMS: {np.sqrt(np.mean(clean_shot**2)):.6f}")
    print(f"Noisy RMS: {np.sqrt(np.mean(noisy_shot**2)):.6f}")
    print(f"Denoised RMS: {np.sqrt(np.mean(denoised_shot**2)):.6f}")
    
    # SNR 계산
    noise = noisy_shot - clean_shot
    snr_before = 20 * np.log10(np.std(clean_shot) / np.std(noise))
    
    residual = denoised_shot - clean_shot
    snr_after = 20 * np.log10(np.std(clean_shot) / np.std(residual))
    
    print(f"\nSNR (노이즈 추가 후): {snr_before:.2f} dB")
    print(f"SNR (노이즈 제거 후): {snr_after:.2f} dB")
    print(f"SNR 개선: {snr_after - snr_before:.2f} dB")
    
    print("\n✅ 전체 워크플로우 완료!")
    print("="*80)
    print("\n💾 생성된 파일:")
    print("  - shot_gather_clean.npz")
    print("  - shot_gather_noisy.npz")
    print("  - shot_gather_denoised.npz")
    print("\n다운로드 방법 (Colab):")
    print("  from google.colab import files")
    print("  files.download('shot_gather_clean.npz')")
    print("  files.download('shot_gather_noisy.npz')")
    print("  files.download('shot_gather_denoised.npz')")


if __name__ == "__main__":
    main()
