#!/usr/bin/env python3
"""
커스텀 해상 탄성파 시뮬레이션 데모
Custom Marine Seismic Simulation Demo

사용자가 파라미터를 조정하여 다양한 시뮬레이션을 실행할 수 있습니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from marine_seismic_simulation import MarineSeismicSimulator


def compare_frequencies():
    """다양한 주파수 대역 비교"""
    print("\n" + "="*70)
    print("다양한 Wavelet 주파수 비교")
    print("="*70)
    
    sim = MarineSeismicSimulator(dt=0.001, nt=2000)
    model = sim.create_synthetic_model(nlayers=5)
    
    frequencies = [20, 30, 40, 50]
    seismic_data = {}
    
    for freq in frequencies:
        seismic = sim.generate_synthetic_seismic(model, freq=freq)
        seismic_data[f'{freq} Hz'] = seismic
        print(f"✓ {freq} Hz 데이터 생성 완료")
    
    # 플롯
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, (title, data) in zip(axes, seismic_data.items()):
        ax.plot(data, sim.time, 'k-', linewidth=0.5)
        ax.fill_betweenx(sim.time, 0, data, where=(data>0), 
                        color='black', alpha=0.5)
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Wavelet Frequency: {title}', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('frequency_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ 주파수 비교 그림 저장: frequency_comparison.png")
    plt.close()


def compare_multiple_strengths():
    """멀티플 강도 비교"""
    print("\n" + "="*70)
    print("멀티플 강도 비교")
    print("="*70)
    
    sim = MarineSeismicSimulator(dt=0.001, nt=2000)
    model = sim.create_synthetic_model(nlayers=5)
    clean = sim.generate_synthetic_seismic(model, freq=30.0)
    
    strengths = [0.0, 0.3, 0.6, 0.9]
    seismic_data = {}
    
    for strength in strengths:
        seismic = sim.add_sea_surface_multiple(clean, model, strength=strength)
        seismic_data[f'Strength: {strength}'] = seismic
        print(f"✓ 멀티플 강도 {strength} 데이터 생성 완료")
    
    # 플롯
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, (title, data) in zip(axes, seismic_data.items()):
        ax.plot(data, sim.time, 'k-', linewidth=0.5)
        ax.fill_betweenx(sim.time, 0, data, where=(data>0), 
                        color='blue', alpha=0.3)
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Multiple {title}', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiple_strength_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ 멀티플 강도 비교 그림 저장: multiple_strength_comparison.png")
    plt.close()


def compare_noise_levels():
    """노이즈 레벨 비교"""
    print("\n" + "="*70)
    print("노이즈 레벨 비교")
    print("="*70)
    
    sim = MarineSeismicSimulator(dt=0.001, nt=2000)
    model = sim.create_synthetic_model(nlayers=5)
    clean = sim.generate_synthetic_seismic(model, freq=30.0)
    
    noise_levels = [0.0, 0.05, 0.10, 0.20]
    seismic_data = {}
    snr_values = {}
    
    for noise_level in noise_levels:
        if noise_level == 0.0:
            noisy = clean
        else:
            noisy = sim.add_marine_noise(clean, noise_level=noise_level)
        
        seismic_data[f'Noise: {noise_level}'] = noisy
        
        # SNR 계산
        if noise_level > 0:
            noise = noisy - clean
            snr = 20 * np.log10(np.std(clean) / np.std(noise))
            snr_values[noise_level] = snr
            print(f"✓ 노이즈 레벨 {noise_level} 데이터 생성 완료 (SNR: {snr:.2f} dB)")
        else:
            print(f"✓ Clean 데이터 (SNR: ∞ dB)")
    
    # 플롯
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, (title, data) in zip(axes, seismic_data.items()):
        ax.plot(data, sim.time, 'k-', linewidth=0.5)
        ax.fill_betweenx(sim.time, 0, data, where=(data>0), 
                        color='red', alpha=0.3)
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Time (s)')
        ax.set_title(title, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_level_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ 노이즈 레벨 비교 그림 저장: noise_level_comparison.png")
    plt.close()


def analyze_multiples():
    """멀티플 상세 분석"""
    print("\n" + "="*70)
    print("멀티플 상세 분석")
    print("="*70)
    
    sim = MarineSeismicSimulator(dt=0.001, nt=2000)
    model = sim.create_synthetic_model(nlayers=5)
    
    clean = sim.generate_synthetic_seismic(model, freq=30.0)
    
    # 각 멀티플 타입 개별 적용
    sea_surface_only = sim.add_sea_surface_multiple(clean, model, strength=0.5)
    internal_only = sim.add_internal_multiples(clean, model, strength=0.3)
    both_multiples = sim.add_sea_surface_multiple(clean, model, strength=0.5)
    both_multiples = sim.add_internal_multiples(both_multiples, model, strength=0.3)
    
    # 플롯
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    data_dict = {
        'Clean (No Multiples)': clean,
        'Sea Surface Multiple Only': sea_surface_only,
        'Internal Multiples Only': internal_only,
        'Both Multiples': both_multiples
    }
    
    for ax, (title, data) in zip(axes, data_dict.items()):
        ax.plot(data, sim.time, 'k-', linewidth=0.5)
        ax.fill_betweenx(sim.time, 0, data, where=(data>0), 
                        color='green', alpha=0.4)
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Time (s)')
        ax.set_title(title, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # RMS 표시
        rms = np.sqrt(np.mean(data**2))
        ax.text(0.05, 0.95, f'RMS: {rms:.4f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('multiple_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ 멀티플 분석 그림 저장: multiple_analysis.png")
    plt.close()
    
    print("\n멀티플 효과:")
    print(f"  Clean RMS: {np.sqrt(np.mean(clean**2)):.6f}")
    print(f"  Sea Surface Multiple RMS: {np.sqrt(np.mean(sea_surface_only**2)):.6f}")
    print(f"  Internal Multiple RMS: {np.sqrt(np.mean(internal_only**2)):.6f}")
    print(f"  Both Multiples RMS: {np.sqrt(np.mean(both_multiples**2)):.6f}")


def create_cmp_gather():
    """CMP(Common Mid-Point) Gather 시뮬레이션"""
    print("\n" + "="*70)
    print("CMP Gather 시뮬레이션")
    print("="*70)
    
    sim = MarineSeismicSimulator(dt=0.001, nt=2000)
    model = sim.create_synthetic_model(nlayers=6)
    
    # 여러 오프셋에 대한 트레이스 생성
    n_traces = 24
    offsets = np.linspace(100, 3000, n_traces)  # 100m ~ 3000m
    
    gather = np.zeros((sim.nt, n_traces))
    
    for i, offset in enumerate(offsets):
        # 오프셋에 따른 NMO(Normal Moveout) 시뮬레이션
        clean = sim.generate_synthetic_seismic(model, freq=30.0)
        
        # 간단한 NMO: t = sqrt(t0^2 + x^2/v^2)
        # 여기서는 단순화를 위해 약간의 시간 지연만 추가
        v_rms = 2500.0  # RMS 속도
        nmo_stretch = np.sqrt(1 + (offset / (v_rms * sim.time[1:] + 1e-6))**2)
        
        # 트레이스에 약간의 변화 추가
        trace = clean * (1 + np.random.normal(0, 0.05))
        
        # 멀티플과 노이즈 추가
        trace = sim.add_sea_surface_multiple(trace, model, strength=0.4)
        trace = sim.add_marine_noise(trace, noise_level=0.05)
        
        gather[:, i] = trace
        
        if (i + 1) % 6 == 0:
            print(f"✓ {i+1}/{n_traces} 트레이스 생성 완료")
    
    # CMP Gather 플롯
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Wiggle traces
    for i in range(n_traces):
        trace = gather[:, i]
        trace_norm = trace / np.max(np.abs(gather)) * 0.8
        x_pos = offsets[i] + trace_norm * 100
        
        ax.plot(x_pos, sim.time, 'k-', linewidth=0.3)
        ax.fill_betweenx(sim.time, offsets[i], x_pos, 
                        where=(trace_norm>0), color='black', alpha=0.6)
    
    ax.set_xlabel('Offset (m)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('CMP Gather (Common Mid-Point)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cmp_gather.png', dpi=300, bbox_inches='tight')
    print("\n✓ CMP Gather 그림 저장: cmp_gather.png")
    plt.close()
    
    # 데이터 저장
    np.savez('cmp_gather_data.npz',
             gather=gather,
             offsets=offsets,
             time=sim.time)
    print("✓ CMP Gather 데이터 저장: cmp_gather_data.npz")


def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("커스텀 해상 탄성파 시뮬레이션 데모")
    print("Custom Marine Seismic Simulation Demo")
    print("="*70)
    
    # 1. 주파수 비교
    compare_frequencies()
    
    # 2. 멀티플 강도 비교
    compare_multiple_strengths()
    
    # 3. 노이즈 레벨 비교
    compare_noise_levels()
    
    # 4. 멀티플 상세 분석
    analyze_multiples()
    
    # 5. CMP Gather 생성
    create_cmp_gather()
    
    print("\n" + "="*70)
    print("✅ 모든 데모 완료!")
    print("="*70)
    print("\n생성된 파일:")
    print("  - frequency_comparison.png")
    print("  - multiple_strength_comparison.png")
    print("  - noise_level_comparison.png")
    print("  - multiple_analysis.png")
    print("  - cmp_gather.png")
    print("  - cmp_gather_data.npz")
    print("="*70)


if __name__ == "__main__":
    main()
