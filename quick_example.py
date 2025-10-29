#!/usr/bin/env python3
"""
빠른 시작 예제
Quick Start Example

해상 탄성파 시뮬레이션의 기본 사용법을 보여줍니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from marine_seismic_simulation import MarineSeismicSimulator


def quick_example():
    """5분 안에 시작하기"""
    print("=" * 70)
    print("해상 탄성파 시뮬레이션 - 빠른 시작")
    print("=" * 70)
    
    # 1. 시뮬레이터 생성
    print("\n[1] 시뮬레이터 생성...")
    sim = MarineSeismicSimulator(dt=0.001, nt=2000)
    
    # 2. 지반 모델 생성
    print("[2] 지반 모델 생성...")
    model = sim.create_synthetic_model(nlayers=5)
    print(f"    생성된 지층 수: {len(model['name'])}")
    print(f"    최대 깊이: {model['depth'][-1] + model['thickness'][-1]:.1f} m")
    
    # 3. Clean 탄성파 생성
    print("[3] Clean 탄성파 데이터 생성...")
    clean = sim.generate_synthetic_seismic(model, freq=30.0)
    print(f"    데이터 길이: {len(clean)} samples")
    print(f"    시간 범위: 0 ~ {sim.time[-1]:.2f} s")
    
    # 4. 멀티플 추가
    print("[4] 멀티플 추가...")
    with_multiples = sim.add_sea_surface_multiple(clean, model, strength=0.5)
    with_multiples = sim.add_internal_multiples(with_multiples, model, strength=0.3)
    
    # 5. 노이즈 추가
    print("[5] 노이즈 추가...")
    noisy = sim.add_marine_noise(with_multiples, noise_level=0.08)
    
    # 6. 시각화
    print("[6] 결과 시각화...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 2x3 그리드
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1행: 탄성파 트레이스
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Clean 탄성파
    ax1.plot(clean, sim.time, 'k-', linewidth=0.5)
    ax1.fill_betweenx(sim.time, 0, clean, where=(clean>0), 
                     color='blue', alpha=0.5)
    ax1.set_title('Clean Seismic', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Amplitude')
    ax1.set_ylabel('Time (s)')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # 멀티플 포함
    ax2.plot(with_multiples, sim.time, 'k-', linewidth=0.5)
    ax2.fill_betweenx(sim.time, 0, with_multiples, where=(with_multiples>0), 
                     color='green', alpha=0.5)
    ax2.set_title('With Multiples', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Amplitude')
    ax2.set_ylabel('Time (s)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    
    # 노이즈 포함
    ax3.plot(noisy, sim.time, 'k-', linewidth=0.5)
    ax3.fill_betweenx(sim.time, 0, noisy, where=(noisy>0), 
                     color='red', alpha=0.5)
    ax3.set_title('With Noise', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Amplitude')
    ax3.set_ylabel('Time (s)')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3)
    
    # 2행: 주파수 스펙트럼
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    freq = np.fft.rfftfreq(len(clean), sim.dt)
    
    # Clean 스펙트럼
    spectrum_clean = np.abs(np.fft.rfft(clean))
    ax4.plot(freq, spectrum_clean, 'b-', linewidth=1.5)
    ax4.set_title('Clean Spectrum', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Amplitude')
    ax4.set_xlim([0, 100])
    ax4.grid(True, alpha=0.3)
    
    # 멀티플 스펙트럼
    spectrum_mult = np.abs(np.fft.rfft(with_multiples))
    ax5.plot(freq, spectrum_mult, 'g-', linewidth=1.5)
    ax5.set_title('With Multiples Spectrum', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Amplitude')
    ax5.set_xlim([0, 100])
    ax5.grid(True, alpha=0.3)
    
    # 노이즈 스펙트럼
    spectrum_noisy = np.abs(np.fft.rfft(noisy))
    ax6.plot(freq, spectrum_noisy, 'r-', linewidth=1.5)
    ax6.set_title('With Noise Spectrum', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Amplitude')
    ax6.set_xlim([0, 100])
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Marine Seismic Survey Simulation - Quick Example', 
                fontsize=16, fontweight='bold')
    
    plt.savefig('quick_example_result.png', dpi=300, bbox_inches='tight')
    print("    결과 저장: quick_example_result.png")
    
    plt.show()
    
    # 7. 통계 출력
    print("\n[7] 통계 정보")
    print(f"    Clean RMS: {np.sqrt(np.mean(clean**2)):.6f}")
    print(f"    멀티플 포함 RMS: {np.sqrt(np.mean(with_multiples**2)):.6f}")
    print(f"    노이즈 포함 RMS: {np.sqrt(np.mean(noisy**2)):.6f}")
    
    noise = noisy - with_multiples
    snr = 20 * np.log10(np.std(with_multiples) / np.std(noise))
    print(f"    SNR: {snr:.2f} dB")
    
    # 8. 데이터 저장
    print("\n[8] 데이터 저장...")
    np.savez('quick_example_data.npz',
             time=sim.time,
             clean=clean,
             with_multiples=with_multiples,
             noisy=noisy,
             model=model)
    print("    데이터 저장: quick_example_data.npz")
    
    print("\n" + "=" * 70)
    print("✅ 완료!")
    print("=" * 70)


def load_and_analyze_example():
    """저장된 데이터 불러오기 예제"""
    print("\n" + "=" * 70)
    print("데이터 로드 및 분석 예제")
    print("=" * 70)
    
    # 데이터 로드
    print("\n[1] 데이터 로드...")
    try:
        data = np.load('quick_example_data.npz', allow_pickle=True)
        
        time = data['time']
        clean = data['clean']
        with_multiples = data['with_multiples']
        noisy = data['noisy']
        model = data['model'].item()
        
        print("    ✓ 데이터 로드 성공")
        print(f"    시간 샘플: {len(time)}")
        print(f"    지층 수: {len(model['name'])}")
        
        # 간단한 분석
        print("\n[2] 신호 분석...")
        print(f"    Clean 최대값: {np.max(np.abs(clean)):.6f}")
        print(f"    Clean 평균값: {np.mean(clean):.6f}")
        print(f"    Noisy 최대값: {np.max(np.abs(noisy)):.6f}")
        print(f"    Noisy 평균값: {np.mean(noisy):.6f}")
        
        # 지층 정보
        print("\n[3] 지층 정보:")
        print(f"    {'Layer':<15} {'Depth (m)':<12} {'Velocity (m/s)'}")
        print("    " + "-" * 50)
        for i in range(min(5, len(model['name']))):
            print(f"    {model['name'][i]:<15} {model['depth'][i]:<12.1f} {model['velocity'][i]:.1f}")
        
        print("\n    ✓ 분석 완료")
        
    except FileNotFoundError:
        print("    ⚠ 데이터 파일을 찾을 수 없습니다.")
        print("    quick_example()를 먼저 실행하세요.")


def main():
    """메인 실행"""
    # 빠른 예제 실행
    quick_example()
    
    # 데이터 로드 예제
    load_and_analyze_example()


if __name__ == "__main__":
    main()
