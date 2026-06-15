"""
测试脚本：验证ECG/PPG处理后的数据

功能：
1. 检查npz文件格式
2. 验证数据shape和类型
3. 时域信号可视化
4. 频域分析（FFT）验证采样率
5. 多片段展示
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class DataValidator:
    """数据验证器"""
    
    def __init__(self, npz_file_path, expected_fs=125, expected_segment_duration=10):
        """
        初始化验证器
        
        Args:
            npz_file_path: npz文件路径
            expected_fs: 期望的采样率（Hz）
            expected_segment_duration: 期望的片段时长（秒）
        """
        self.npz_path = Path(npz_file_path)
        self.expected_fs = expected_fs
        self.expected_segment_duration = expected_segment_duration
        self.expected_segment_length = expected_fs * expected_segment_duration
        
        # 加载数据
        self.data = None
        self.load_data()
    
    def load_data(self):
        """加载npz文件"""
        print(f"Loading data from: {self.npz_path}")
        try:
            self.data = np.load(self.npz_path, allow_pickle=True)
            print("✓ Data loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load data: {e}")
            raise
    
    def check_format(self):
        """检查数据格式"""
        print("\n" + "="*60)
        print("1. 检查数据格式")
        print("="*60)
        
        # 检查必需的键
        required_keys = ['file_name', 'PPG', 'ECG']
        optional_keys = ['ECG_I']
        
        print("\n必需的键:")
        for key in required_keys:
            if key in self.data:
                print(f"  ✓ {key} 存在")
            else:
                print(f"  ✗ {key} 缺失！")
                return False
        
        print("\n可选的键:")
        for key in optional_keys:
            if key in self.data:
                print(f"  ✓ {key} 存在")
            else:
                print(f"  - {key} 不存在")
        
        # 打印file_name
        file_name = self.data['file_name']
        if isinstance(file_name, np.ndarray):
            file_name = str([file_name[0]])
        print(f"\nSubject ID: {file_name}")
        
        return True
    
    def check_shape_and_type(self):
        """检查数据shape和类型"""
        print("\n" + "="*60)
        print("2. 检查数据Shape和类型")
        print("="*60)
        
        ppg = self.data['PPG']
        ecg = self.data['ECG']
        
        print(f"\nPPG:")
        print(f"  Shape: {ppg.shape}")
        print(f"  Dtype: {ppg.dtype}")
        print(f"  Range: [{ppg.min():.4f}, {ppg.max():.4f}]")
        print(f"  Mean: {ppg.mean():.4f}, Std: {ppg.std():.4f}")
        
        print(f"\nECG (Lead II):")
        print(f"  Shape: {ecg.shape}")
        print(f"  Dtype: {ecg.dtype}")
        print(f"  Range: [{ecg.min():.4f}, {ecg.max():.4f}]")
        print(f"  Mean: {ecg.mean():.4f}, Std: {ecg.std():.4f}")
        
        if 'ECG_I' in self.data:
            ecg_i = self.data['ECG_I']
            print(f"\nECG_I (Lead I):")
            print(f"  Shape: {ecg_i.shape}")
            print(f"  Dtype: {ecg_i.dtype}")
            print(f"  Range: [{ecg_i.min():.4f}, {ecg_i.max():.4f}]")
            print(f"  Mean: {ecg_i.mean():.4f}, Std: {ecg_i.std():.4f}")
        
        # 验证shape
        print("\n验证Shape:")
        n_segments = ppg.shape[0]
        segment_length = ppg.shape[1]
        
        print(f"  片段数量: {n_segments}")
        print(f"  每个片段长度: {segment_length}")
        print(f"  期望片段长度: {self.expected_segment_length}")
        
        if segment_length == self.expected_segment_length:
            print(f"  ✓ 片段长度正确 ({self.expected_segment_duration}秒 × {self.expected_fs}Hz = {self.expected_segment_length}点)")
        else:
            print(f"  ✗ 片段长度不匹配！期望{self.expected_segment_length}，实际{segment_length}")
        
        # 检查所有信号的片段数是否一致
        if ecg.shape[0] == n_segments:
            print(f"  ✓ PPG和ECG片段数一致")
        else:
            print(f"  ✗ PPG和ECG片段数不一致！")
        
        if 'ECG_I' in self.data and self.data['ECG_I'].shape[0] == n_segments:
            print(f"  ✓ ECG_I片段数一致")
        
        return True
    
    def plot_time_domain(self, segment_idx=0, num_segments=3):
        """绘制时域信号"""
        print("\n" + "="*60)
        print("3. 时域信号可视化")
        print("="*60)
        
        ppg = self.data['PPG']
        ecg = self.data['ECG']
        n_segments = ppg.shape[0]
        
        # 确保不超过实际片段数
        num_segments = min(num_segments, n_segments)
        
        # 确定是否有ECG_I
        has_ecg_i = 'ECG_I' in self.data
        n_rows = 3 if has_ecg_i else 2
        
        fig, axes = plt.subplots(n_rows, num_segments, 
                                figsize=(6*num_segments, 3*n_rows))
        
        if num_segments == 1:
            axes = axes.reshape(-1, 1)
        
        time_axis = np.arange(self.expected_segment_length) / self.expected_fs
        
        for i in range(num_segments):
            idx = segment_idx + i
            if idx >= n_segments:
                break
            
            # PPG
            axes[0, i].plot(time_axis, ppg[idx], 'b-', linewidth=0.5)
            axes[0, i].set_title(f'PPG - Segment {idx}')
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel('Amplitude')
            axes[0, i].grid(True, alpha=0.3)
            
            # ECG (Lead II)
            axes[1, i].plot(time_axis, ecg[idx], 'r-', linewidth=0.5)
            axes[1, i].set_title(f'ECG (Lead II) - Segment {idx}')
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel('Amplitude')
            axes[1, i].grid(True, alpha=0.3)
            
            # ECG_I (Lead I) if available
            if has_ecg_i:
                ecg_i = self.data['ECG_I']
                axes[2, i].plot(time_axis, ecg_i[idx], 'g-', linewidth=0.5)
                axes[2, i].set_title(f'ECG_I (Lead I) - Segment {idx}')
                axes[2, i].set_xlabel('Time (s)')
                axes[2, i].set_ylabel('Amplitude')
                axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.npz_path.parent / f"{self.npz_path.stem}_time_domain.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 时域图保存至: {output_path}")
        
        return fig
    
    def plot_frequency_domain(self, segment_idx=0):
        """绘制频域信号（FFT分析）"""
        print("\n" + "="*60)
        print("4. 频域分析（验证采样率）")
        print("="*60)
        
        ppg = self.data['PPG']
        ecg = self.data['ECG']
        
        # 确定是否有ECG_I
        has_ecg_i = 'ECG_I' in self.data
        n_rows = 3 if has_ecg_i else 2
        
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4*n_rows))
        if not has_ecg_i:
            axes = [axes[0], axes[1]]
        
        # 计算频率轴
        n_fft = self.expected_segment_length
        freq = np.fft.rfftfreq(n_fft, 1/self.expected_fs)
        
        # PPG FFT
        ppg_fft = np.abs(np.fft.rfft(ppg[segment_idx]))
        axes[0].semilogy(freq, ppg_fft, 'b-', linewidth=0.5)
        axes[0].set_title(f'PPG Frequency Spectrum - Segment {segment_idx}')
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Magnitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=self.expected_fs/2, color='r', linestyle='--', 
                       label=f'Nyquist Freq ({self.expected_fs/2} Hz)')
        axes[0].legend()
        axes[0].set_xlim([0, min(50, self.expected_fs/2)])
        
        # ECG FFT
        ecg_fft = np.abs(np.fft.rfft(ecg[segment_idx]))
        axes[1].semilogy(freq, ecg_fft, 'r-', linewidth=0.5)
        axes[1].set_title(f'ECG (Lead II) Frequency Spectrum - Segment {segment_idx}')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Magnitude')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=self.expected_fs/2, color='r', linestyle='--', 
                       label=f'Nyquist Freq ({self.expected_fs/2} Hz)')
        axes[1].legend()
        axes[1].set_xlim([0, min(50, self.expected_fs/2)])
        
        # ECG_I FFT if available
        if has_ecg_i:
            ecg_i = self.data['ECG_I']
            ecg_i_fft = np.abs(np.fft.rfft(ecg_i[segment_idx]))
            axes[2].semilogy(freq, ecg_i_fft, 'g-', linewidth=0.5)
            axes[2].set_title(f'ECG_I (Lead I) Frequency Spectrum - Segment {segment_idx}')
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Magnitude')
            axes[2].grid(True, alpha=0.3)
            axes[2].axvline(x=self.expected_fs/2, color='r', linestyle='--', 
                           label=f'Nyquist Freq ({self.expected_fs/2} Hz)')
            axes[2].legend()
            axes[2].set_xlim([0, min(50, self.expected_fs/2)])
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.npz_path.parent / f"{self.npz_path.stem}_frequency_domain.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 频域图保存至: {output_path}")
        
        # 分析主要频率成分
        print("\n主要频率成分分析:")
        
        # PPG
        ppg_peak_idx = np.argsort(ppg_fft)[-5:][::-1]
        ppg_peak_freqs = freq[ppg_peak_idx]
        print(f"\nPPG 前5个主要频率: {ppg_peak_freqs[:5]} Hz")
        print(f"  (典型心率范围: 0.8-2.0 Hz, 即48-120 bpm)")
        
        # ECG
        ecg_peak_idx = np.argsort(ecg_fft)[-5:][::-1]
        ecg_peak_freqs = freq[ecg_peak_idx]
        print(f"\nECG (Lead II) 前5个主要频率: {ecg_peak_freqs[:5]} Hz")
        print(f"  (典型心率范围: 0.8-2.0 Hz, 即48-120 bpm)")
        
        # ECG_I
        if has_ecg_i:
            ecg_i = self.data['ECG_I']
            ecg_i_fft = np.abs(np.fft.rfft(ecg_i[segment_idx]))
            ecg_i_peak_idx = np.argsort(ecg_i_fft)[-5:][::-1]
            ecg_i_peak_freqs = freq[ecg_i_peak_idx]
            print(f"\nECG_I (Lead I) 前5个主要频率: {ecg_i_peak_freqs[:5]} Hz")
            print(f"  (典型心率范围: 0.8-2.0 Hz, 即48-120 bpm)")
        
        return fig
    
    def plot_comparison(self, segment_idx=0):
        """绘制PPG、ECG和ECG_I的对比图"""
        print("\n" + "="*60)
        print("5. PPG与ECG对比")
        print("="*60)
        
        ppg = self.data['PPG'][segment_idx]
        ecg = self.data['ECG'][segment_idx]
        
        # 确定是否有ECG_I
        has_ecg_i = 'ECG_I' in self.data
        
        # 归一化到[0, 1]便于对比
        ppg_norm = (ppg - ppg.min()) / (ppg.max() - ppg.min())
        ecg_norm = (ecg - ecg.min()) / (ecg.max() - ecg.min())
        
        # ppg_norm = ppg
        # ecg_norm = ecg

        time_axis = np.arange(len(ppg)) / self.expected_fs
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        
        ax.plot(time_axis, ppg_norm, 'b-', linewidth=1, label='PPG (normalized)', alpha=0.7)
        ax.plot(time_axis, ecg_norm, 'r-', linewidth=1, label='ECG Lead II (normalized)', alpha=0.7)
        
        # 如果有ECG_I，也添加到对比图中
        if has_ecg_i:
            ecg_i = self.data['ECG_I'][segment_idx]
            ecg_i_norm = (ecg_i - ecg_i.min()) / (ecg_i.max() - ecg_i.min())
            ax.plot(time_axis, ecg_i_norm, 'g-', linewidth=1, label='ECG_I Lead I (normalized)', alpha=0.7)
        
        ax.set_title(f'PPG vs ECG Comparison - Segment {segment_idx}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = self.npz_path.parent / f"{self.npz_path.stem}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 对比图保存至: {output_path}")
        
        return fig
    
    def run_all_tests(self, show_plots=True):
        """运行所有测试"""
        print("\n" + "="*80)
        print("开始数据验证")
        print("="*80)
        
        # 1. 检查格式
        if not self.check_format():
            print("\n✗ 格式检查失败！")
            return False
        
        # 2. 检查shape和类型
        if not self.check_shape_and_type():
            print("\n✗ Shape和类型检查失败！")
            return False
        
        # 3. 时域可视化
        self.plot_time_domain(segment_idx=0, num_segments=3)
        
        # 4. 频域分析
        self.plot_frequency_domain(segment_idx=0)
        
        # 5. PPG-ECG对比
        self.plot_comparison(segment_idx=0)
        
        print("\n" + "="*80)
        print("✓ 所有测试完成！")
        print("="*80)
        
        if show_plots:
            plt.show()
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Test and validate processed ECG/PPG data in npz format'
    )
    parser.add_argument(
        '--npz_file',
        type=str,
        required=True,
        help='Path to the npz file to validate'
    )
    parser.add_argument(
        '--fs',
        type=int,
        default=125,
        help='Expected sampling rate in Hz (default: 125)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Expected segment duration in seconds (default: 10)'
    )
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='Do not display plots (only save to files)'
    )
    
    args = parser.parse_args()
    
    # 创建验证器并运行测试
    validator = DataValidator(
        npz_file_path=args.npz_file,
        expected_fs=args.fs,
        expected_segment_duration=args.duration
    )
    
    validator.run_all_tests(show_plots=not args.no_show)


if __name__ == '__main__':
    main()


'''
使用示例：

# 基本用法
python pl_test.py --npz_file /path/to/p10014354.npz

# 指定参数
python pl_test.py \
    --npz_file /path/to/p10014354.npz \
    --fs 125 \
    --duration 10

# 不显示图像（只保存）
python pl_test.py \
    --npz_file /path/to/p10014354.npz \
    --no_show

python pl_test.py \
    --npz_file /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered/p10019003.npz \
    --fs 125 \
    --duration 10 \
    --no_show
'''