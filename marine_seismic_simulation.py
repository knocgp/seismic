#!/usr/bin/env python3
"""
해상 탄성파 탐사 시뮬레이션
Marine Seismic Survey Simulation

이 프로그램은 다음을 포함합니다:
1. 합성 지반 모델 생성
2. 합성 탄성파 데이터 생성 (Reflectivity Method)
3. 해상 탄성파 특유의 노이즈 추가
4. 멀티플(Multiples) 추가
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class MarineSeismicSimulator:
    """해상 탄성파 탐사 시뮬레이터"""
    
    def __init__(self, dt: float = 0.001, nt: int = 2000):
        """
        초기화
        
        Parameters:
        -----------
        dt : float
            샘플링 간격 (초)
        nt : int
            시간 샘플 개수
        """
        self.dt = dt
        self.nt = nt
        self.time = np.arange(nt) * dt
        
    def create_synthetic_model(self, nlayers: int = 5) -> Dict:
        """
        합성 지반 모델 생성
        
        Parameters:
        -----------
        nlayers : int
            지층 개수
            
        Returns:
        --------
        model : dict
            velocity : 속도 (m/s)
            density : 밀도 (kg/m³)
            thickness : 두께 (m)
            depth : 깊이 (m)
        """
        # 해수층 포함 (첫 번째 층)
        model = {
            'velocity': [],
            'density': [],
            'thickness': [],
            'depth': [],
            'name': []
        }
        
        # 1. 해수층 (Water layer)
        model['velocity'].append(1500.0)  # 해수 속도
        model['density'].append(1030.0)   # 해수 밀도
        model['thickness'].append(500.0)  # 500m 수심
        model['depth'].append(0.0)
        model['name'].append('Water')
        
        # 2. 해저면 (Seabed) - 연약한 퇴적층
        model['velocity'].append(1800.0)
        model['density'].append(2000.0)
        model['thickness'].append(300.0)
        model['depth'].append(500.0)
        model['name'].append('Soft sediment')
        
        # 3-N. 지하 지층들 (점진적으로 증가하는 속도와 밀도)
        current_depth = 800.0
        for i in range(nlayers - 2):
            # 속도: 2000 ~ 4500 m/s
            vp = 2000.0 + (i * 500.0) + np.random.normal(0, 100)
            # 밀도: 2100 ~ 2600 kg/m³
            rho = 2100.0 + (i * 100.0) + np.random.normal(0, 50)
            # 두께: 200 ~ 500 m
            thickness = 200.0 + np.random.uniform(100, 300)
            
            model['velocity'].append(vp)
            model['density'].append(rho)
            model['thickness'].append(thickness)
            model['depth'].append(current_depth)
            model['name'].append(f'Layer {i+3}')
            
            current_depth += thickness
        
        return model
    
    def calculate_reflection_coefficients(self, model: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        반사 계수 계산 (Reflection Coefficients)
        
        Parameters:
        -----------
        model : dict
            지층 모델
            
        Returns:
        --------
        rc : np.ndarray
            반사 계수
        times : np.ndarray
            반사 시간 (초)
        """
        velocities = np.array(model['velocity'])
        densities = np.array(model['density'])
        thicknesses = np.array(model['thickness'])
        
        # 음향 임피던스 (Acoustic Impedance)
        impedance = velocities * densities
        
        # 반사 계수 계산
        rc = np.zeros(len(velocities) - 1)
        for i in range(len(velocities) - 1):
            rc[i] = (impedance[i+1] - impedance[i]) / (impedance[i+1] + impedance[i])
        
        # 반사 시간 계산 (양방향 주시)
        times = np.zeros(len(velocities) - 1)
        cumulative_time = 0
        for i in range(len(velocities) - 1):
            # 층을 통과하는 시간
            travel_time = thicknesses[i] / velocities[i]
            cumulative_time += travel_time
            times[i] = cumulative_time * 2  # 양방향
        
        return rc, times
    
    def ricker_wavelet(self, freq: float = 30.0) -> np.ndarray:
        """
        Ricker 파형 생성 (주파수 중심의 wavelet)
        
        Parameters:
        -----------
        freq : float
            중심 주파수 (Hz)
            
        Returns:
        --------
        wavelet : np.ndarray
            Ricker wavelet
        """
        duration = 0.2  # 200 ms
        t = np.arange(-duration/2, duration/2, self.dt)
        
        # Ricker wavelet formula
        a = (np.pi * freq * t) ** 2
        wavelet = (1 - 2*a) * np.exp(-a)
        
        # 정규화
        wavelet = wavelet / np.max(np.abs(wavelet))
        
        return wavelet
    
    def generate_synthetic_seismic(self, model: Dict, freq: float = 30.0) -> np.ndarray:
        """
        합성 탄성파 데이터 생성
        
        Parameters:
        -----------
        model : dict
            지층 모델
        freq : float
            Wavelet 중심 주파수
            
        Returns:
        --------
        seismic : np.ndarray
            합성 탄성파 데이터
        """
        # 반사 계수 및 시간 계산
        rc, reflection_times = self.calculate_reflection_coefficients(model)
        
        # Reflectivity series 생성
        reflectivity = np.zeros(self.nt)
        for rc_val, t_refl in zip(rc, reflection_times):
            idx = int(t_refl / self.dt)
            if idx < self.nt:
                reflectivity[idx] = rc_val
        
        # Wavelet 생성
        wavelet = self.ricker_wavelet(freq)
        
        # 컨볼루션으로 합성 데이터 생성
        seismic = signal.convolve(reflectivity, wavelet, mode='same')
        
        return seismic
    
    def add_sea_surface_multiple(self, seismic: np.ndarray, model: Dict, 
                                  strength: float = 0.5) -> np.ndarray:
        """
        해면 멀티플 추가 (Sea Surface Multiple)
        해수면에서 반사되어 다시 들어오는 신호
        
        Parameters:
        -----------
        seismic : np.ndarray
            원본 탄성파 데이터
        model : dict
            지층 모델
        strength : float
            멀티플 강도 (0~1)
            
        Returns:
        --------
        seismic_with_multiple : np.ndarray
            멀티플이 추가된 데이터
        """
        # 해수 깊이
        water_depth = model['thickness'][0]
        water_velocity = model['velocity'][0]
        
        # 해수층 양방향 주시 시간
        two_way_time = 2 * water_depth / water_velocity
        delay_samples = int(two_way_time / self.dt)
        
        # 해면 반사 계수 (공기-물 경계, 거의 -1)
        sea_surface_rc = -0.95
        
        # 멀티플 생성 (1차 멀티플)
        multiple = np.zeros_like(seismic)
        if delay_samples < len(seismic):
            multiple[delay_samples:] = seismic[:-delay_samples] * sea_surface_rc * strength
        
        # 2차 멀티플도 추가 (더 약하게)
        if 2 * delay_samples < len(seismic):
            multiple[2*delay_samples:] += seismic[:-2*delay_samples] * (sea_surface_rc**2) * strength * 0.5
        
        return seismic + multiple
    
    def add_internal_multiples(self, seismic: np.ndarray, model: Dict,
                               strength: float = 0.3) -> np.ndarray:
        """
        내부 멀티플 추가 (Internal Multiples)
        지층 경계면에서의 다중 반사
        
        Parameters:
        -----------
        seismic : np.ndarray
            원본 탄성파 데이터
        model : dict
            지층 모델
        strength : float
            멀티플 강도
            
        Returns:
        --------
        seismic_with_multiple : np.ndarray
            내부 멀티플이 추가된 데이터
        """
        result = seismic.copy()
        
        # 주요 경계면에서의 내부 멀티플 시뮬레이션
        rc, reflection_times = self.calculate_reflection_coefficients(model)
        
        # 강한 반사면 찾기 (|RC| > 0.1)
        strong_reflectors = [(t, rc_val) for t, rc_val in zip(reflection_times, rc) 
                           if abs(rc_val) > 0.1]
        
        # 두 개의 강한 반사면 사이에서 멀티플 생성
        for i, (t1, rc1) in enumerate(strong_reflectors):
            for t2, rc2 in strong_reflectors[i+1:]:
                # 멀티플 지연 시간
                multiple_delay = t2 - t1 + (t2 - t1)  # 추가 왕복
                delay_samples = int(multiple_delay / self.dt)
                
                if delay_samples < len(seismic):
                    # 멀티플 강도는 두 반사계수의 곱
                    multiple_strength = rc1 * rc2 * strength
                    result[delay_samples:] += seismic[:-delay_samples] * multiple_strength
        
        return result
    
    def add_marine_noise(self, seismic: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        해상 탄성파 특유의 노이즈 추가
        
        Parameters:
        -----------
        seismic : np.ndarray
            원본 탄성파 데이터
        noise_level : float
            노이즈 레벨 (신호 대비)
            
        Returns:
        --------
        noisy_seismic : np.ndarray
            노이즈가 추가된 데이터
        """
        result = seismic.copy()
        signal_power = np.std(seismic)
        
        # 1. 백색 잡음 (White noise) - 전자 장비 노이즈
        white_noise = np.random.normal(0, noise_level * signal_power * 0.3, len(seismic))
        result += white_noise
        
        # 2. 선박 노이즈 (Ship noise) - 저주파 노이즈
        ship_freq = np.random.uniform(2, 8)  # 2-8 Hz
        ship_noise = noise_level * signal_power * 0.5 * np.sin(2 * np.pi * ship_freq * self.time)
        # 시간에 따라 변동
        ship_noise *= (1 + 0.3 * np.sin(2 * np.pi * 0.5 * self.time))
        result += ship_noise
        
        # 3. 해양 환경 노이즈 (Ocean ambient noise)
        # 저주파 대역 (1-20 Hz)
        ocean_noise = np.zeros_like(seismic)
        for freq in np.random.uniform(1, 20, 5):
            amplitude = noise_level * signal_power * 0.2 * np.random.uniform(0.5, 1.0)
            ocean_noise += amplitude * np.sin(2 * np.pi * freq * self.time + 
                                             np.random.uniform(0, 2*np.pi))
        result += ocean_noise
        
        # 4. 스웰 노이즈 (Swell noise) - 해수면 파도
        swell_freq = np.random.uniform(0.1, 0.5)  # 0.1-0.5 Hz
        swell_noise = noise_level * signal_power * 0.4 * np.sin(2 * np.pi * swell_freq * self.time)
        # 진폭 변조
        swell_noise *= (1 + 0.5 * np.sin(2 * np.pi * 0.2 * self.time))
        result += swell_noise
        
        # 5. 버스트 노이즈 (Burst noise) - 간헐적 충격
        num_bursts = np.random.randint(3, 8)
        for _ in range(num_bursts):
            burst_time = np.random.randint(0, len(seismic))
            burst_duration = np.random.randint(10, 50)
            if burst_time + burst_duration < len(seismic):
                burst = noise_level * signal_power * 2.0 * np.random.randn(burst_duration)
                result[burst_time:burst_time+burst_duration] += burst
        
        return result
    
    def plot_model(self, model: Dict, save_path: str = None):
        """지층 모델 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        depths = model['depth']
        velocities = model['velocity']
        densities = model['density']
        
        # 속도 프로파일
        for i in range(len(depths)):
            depth_top = depths[i]
            depth_bottom = depths[i] + model['thickness'][i]
            
            # 층 그리기
            ax1.fill_between([velocities[i]-100, velocities[i]+100], 
                            depth_top, depth_bottom, 
                            alpha=0.3, label=model['name'][i] if i < 5 else None)
            ax1.plot([velocities[i], velocities[i]], [depth_top, depth_bottom], 
                    'b-', linewidth=2)
            
            # 밀도 그리기
            ax2.fill_between([densities[i]-50, densities[i]+50], 
                            depth_top, depth_bottom, 
                            alpha=0.3)
            ax2.plot([densities[i], densities[i]], [depth_top, depth_bottom], 
                    'r-', linewidth=2)
        
        ax1.set_xlabel('Velocity (m/s)', fontsize=12)
        ax1.set_ylabel('Depth (m)', fontsize=12)
        ax1.set_title('Velocity Model', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlabel('Density (kg/m³)', fontsize=12)
        ax2.set_ylabel('Depth (m)', fontsize=12)
        ax2.set_title('Density Model', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"모델 그림 저장: {save_path}")
        
        plt.show()
    
    def plot_seismic_comparison(self, seismic_data: Dict, save_path: str = None):
        """여러 탄성파 데이터 비교 플롯"""
        n_traces = len(seismic_data)
        fig, axes = plt.subplots(1, n_traces, figsize=(5*n_traces, 10))
        
        if n_traces == 1:
            axes = [axes]
        
        for ax, (title, data) in zip(axes, seismic_data.items()):
            # Wiggle trace plot
            ax.plot(data, self.time, 'k-', linewidth=0.5)
            ax.fill_betweenx(self.time, 0, data, where=(data>0), 
                            color='black', alpha=0.5)
            
            ax.set_xlabel('Amplitude', fontsize=11)
            ax.set_ylabel('Time (s)', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-np.max(np.abs(data))*1.1, np.max(np.abs(data))*1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"탄성파 비교 그림 저장: {save_path}")
        
        plt.show()
    
    def plot_spectrum(self, seismic_data: Dict, save_path: str = None):
        """주파수 스펙트럼 분석"""
        fig, axes = plt.subplots(len(seismic_data), 1, figsize=(12, 4*len(seismic_data)))
        
        if len(seismic_data) == 1:
            axes = [axes]
        
        for ax, (title, data) in zip(axes, seismic_data.items()):
            # FFT 계산
            freq = np.fft.rfftfreq(len(data), self.dt)
            spectrum = np.abs(np.fft.rfft(data))
            
            ax.plot(freq, spectrum, 'b-', linewidth=1.5)
            ax.set_xlabel('Frequency (Hz)', fontsize=11)
            ax.set_ylabel('Amplitude', fontsize=11)
            ax.set_title(f'Frequency Spectrum - {title}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 100])  # 0-100 Hz 범위
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"스펙트럼 그림 저장: {save_path}")
        
        plt.show()


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("해상 탄성파 탐사 시뮬레이션")
    print("Marine Seismic Survey Simulation")
    print("=" * 70)
    print()
    
    # 시뮬레이터 초기화
    simulator = MarineSeismicSimulator(dt=0.001, nt=2000)
    print("✓ 시뮬레이터 초기화 완료")
    
    # 1. 합성 지반 모델 생성
    print("\n[1단계] 합성 지반 모델 생성...")
    model = simulator.create_synthetic_model(nlayers=6)
    
    print("\n지층 정보:")
    print(f"{'Layer':<15} {'Depth (m)':<12} {'Thickness (m)':<15} {'Velocity (m/s)':<15} {'Density (kg/m³)'}")
    print("-" * 75)
    for i in range(len(model['name'])):
        print(f"{model['name'][i]:<15} {model['depth'][i]:<12.1f} {model['thickness'][i]:<15.1f} "
              f"{model['velocity'][i]:<15.1f} {model['density'][i]:<15.1f}")
    
    # 모델 시각화
    simulator.plot_model(model, save_path='marine_seismic_model.png')
    
    # 2. 합성 탄성파 데이터 생성
    print("\n[2단계] 합성 탄성파 데이터 생성...")
    clean_seismic = simulator.generate_synthetic_seismic(model, freq=30.0)
    print("✓ Clean 탄성파 데이터 생성 완료")
    
    # 3. 멀티플 추가
    print("\n[3단계] 멀티플 추가...")
    seismic_with_multiples = simulator.add_sea_surface_multiple(clean_seismic, model, strength=0.5)
    seismic_with_multiples = simulator.add_internal_multiples(seismic_with_multiples, model, strength=0.3)
    print("✓ 해면 멀티플 추가 완료")
    print("✓ 내부 멀티플 추가 완료")
    
    # 4. 노이즈 추가
    print("\n[4단계] 해상 노이즈 추가...")
    noisy_seismic = simulator.add_marine_noise(seismic_with_multiples, noise_level=0.08)
    print("✓ 백색 잡음 추가")
    print("✓ 선박 노이즈 추가")
    print("✓ 해양 환경 노이즈 추가")
    print("✓ 스웰 노이즈 추가")
    print("✓ 버스트 노이즈 추가")
    
    # 5. 결과 비교
    print("\n[5단계] 결과 시각화...")
    seismic_comparison = {
        'Clean Seismic': clean_seismic,
        'With Multiples': seismic_with_multiples,
        'With Noise': noisy_seismic
    }
    
    simulator.plot_seismic_comparison(seismic_comparison, 
                                     save_path='marine_seismic_comparison.png')
    
    # 6. 주파수 스펙트럼 분석
    print("\n[6단계] 주파수 스펙트럼 분석...")
    simulator.plot_spectrum(seismic_comparison, 
                           save_path='marine_seismic_spectrum.png')
    
    # 7. 데이터 저장
    print("\n[7단계] 데이터 저장...")
    np.savez('marine_seismic_data.npz',
             time=simulator.time,
             clean=clean_seismic,
             with_multiples=seismic_with_multiples,
             noisy=noisy_seismic,
             model=model)
    print("✓ 데이터 저장 완료: marine_seismic_data.npz")
    
    # 통계 정보
    print("\n" + "=" * 70)
    print("데이터 통계")
    print("=" * 70)
    print(f"샘플링 간격: {simulator.dt*1000:.2f} ms")
    print(f"총 시간: {simulator.time[-1]:.2f} s")
    print(f"Clean 신호 RMS: {np.sqrt(np.mean(clean_seismic**2)):.6f}")
    print(f"멀티플 포함 RMS: {np.sqrt(np.mean(seismic_with_multiples**2)):.6f}")
    print(f"노이즈 포함 RMS: {np.sqrt(np.mean(noisy_seismic**2)):.6f}")
    print(f"SNR: {20*np.log10(np.std(seismic_with_multiples)/np.std(noisy_seismic-seismic_with_multiples)):.2f} dB")
    
    print("\n✅ 시뮬레이션 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
