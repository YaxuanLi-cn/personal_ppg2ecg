"""
ECG/PPG Data Processing Pipeline - Modified Version

处理流程：
1. 读取Arrow文件中的ECG和PPG数据
2. 计算Lead I（可选）和Lead II
3. 重采样到125Hz
4. 切分成10秒片段
5. 按subject_id保存为npz文件
"""

import os
import traceback
import numpy as np
from pathlib import Path
from scipy import signal
import pyarrow as pa
import pyarrow.ipc as ipc
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ECGLeadCalculator:
    """
    ECG导联计算器
    
    标准12导联关系：
    - Lead I = LA - RA
    - Lead II = LL - RA  
    - Lead III = LL - LA
    """
    
    @staticmethod
    def calculate_lead_i(ecg_data, ecg_names):
        """
        计算或获取Lead I
        
        优先级：
        1. 直接使用 'I' 或 'Lead I'
        """
        # 规范化导联名称
        names_upper = [name.upper().replace('LEAD', '').replace('_', '').strip() 
                       for name in ecg_names]

        lead_indices = {}
        for idx, name in enumerate(names_upper):
            if name in ['I', 'LEADI']:
                lead_indices['I'] = idx
            elif name in ['II', 'LEADII']:
                lead_indices['II'] = idx
            elif name in ['III', 'LEADIII']:
                lead_indices['III'] = idx
        
        # 直接返回Lead I
        if 'I' in lead_indices:
            return ecg_data[lead_indices['I']]
        
        return None
    
    @staticmethod
    def calculate_lead_ii(ecg_data, ecg_names):
        """
        计算或获取Lead II
        
        优先级：
        1. 直接使用 'II' 或 'Lead II'
        """
        # 规范化导联名称
        names_upper = [name.upper().replace('LEAD', '').replace('_', '').strip() 
                       for name in ecg_names]

        lead_indices = {}
        for idx, name in enumerate(names_upper):
            if name in ['I', 'LEADI']:
                lead_indices['I'] = idx
            elif name in ['II', 'LEADII']:
                lead_indices['II'] = idx
            elif name in ['III', 'LEADIII']:
                lead_indices['III'] = idx
        
        # 直接返回Lead II
        if 'II' in lead_indices:
            return ecg_data[lead_indices['II']]
        
        return None
    
    @staticmethod
    def calculate_lead_avr(ecg_data, ecg_names):
        """
        计算或获取Lead aVR
        
        优先级：
        1. 直接使用 'aVR' 或 'Lead aVR'
        """
        # 规范化导联名称
        names_upper = [name.upper().replace('LEAD', '').replace('_', '').strip() 
                       for name in ecg_names]

        lead_indices = {}
        for idx, name in enumerate(names_upper):
            if name in ['AVR', 'LEADA']:
                lead_indices['AVR'] = idx
            elif name in ['I', 'LEADI']:
                lead_indices['I'] = idx
            elif name in ['II', 'LEADII']:
                lead_indices['II'] = idx
            elif name in ['III', 'LEADIII']:
                lead_indices['III'] = idx

        # 直接返回Lead aVR
        if 'AVR' in lead_indices:
            return ecg_data[lead_indices['AVR']]
        
        return None


class SignalProcessor:
    """信号处理类：重采样"""
    
    @staticmethod
    def resample_signal(signal_data, original_fs, target_fs=125):
        """
        重采样信号到目标采样率
        
        Args:
            signal_data: 原始信号数据
            original_fs: 原始采样率
            target_fs: 目标采样率（默认125Hz）
        
        Returns:
            重采样后的信号
        """
        if original_fs == target_fs:
            return signal_data
        
        # 计算新的采样点数
        original_length = len(signal_data)
        new_length = int(original_length * target_fs / original_fs)
        
        # 使用scipy的resample进行重采样
        resampled_signal = signal.resample(signal_data, new_length)
        
        return resampled_signal


class ECGPPGPipeline:
    """ECG/PPG数据处理主pipeline"""
    
    def __init__(self, input_dir, output_dir, target_fs=125, segment_duration=10, 
                 process_lead_i=False, process_lead_avr=False):
        """
        初始化pipeline
        
        Args:
            input_dir: 输入数据目录（包含多个shard）
            output_dir: 输出目录
            target_fs: 目标采样率（默认125Hz）
            segment_duration: 片段时长（秒，默认10秒）
            process_lead_i: 是否处理Lead I（默认False）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_fs = target_fs
        self.segment_duration = segment_duration
        self.process_lead_i = process_lead_i
        self.process_lead_avr = process_lead_avr
        self.segment_length = target_fs * segment_duration  # 10秒的采样点数
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于存储每个subject的数据
        self.subject_data = defaultdict(list)
        
        self.lead_calculator = ECGLeadCalculator()
        self.signal_processor = SignalProcessor()
    
    def extract_subject_id(self, record_name):
        """
        从record name中提取subject_id
        
        例如: p100/p10014354/81739927/81739927_0016_seg0003 -> p10014354
        """
        parts = record_name.split('/')
        if len(parts) >= 2:
            return parts[1]
        return None
    
    def process_record(self, record):
        """
        处理单个record
        
        Args:
            record: Arrow文件中的一条记录
        
        Returns:
            处理后的数据列表，每个元素是一个10秒片段
        """
        try:
            # 提取数据
            ecg_data = np.array(record['ecg'])  # shape: (n_leads, n_samples)
            ecg_names = record['ecg_names']
            ecg_fs = record['ecg_fs']
            
            ppg_data = np.array(record['ppg'])  # shape: (n_channels, n_samples)
            ppg_fs = record['ppg_fs']
            
            # 计算Lead II（必须）
            lead_ii = self.lead_calculator.calculate_lead_ii(ecg_data, ecg_names)
            
            if lead_ii is None:
                print(f"Warning: Cannot calculate Lead II for record {record['record_name']}")
                print(f"  Available leads: {ecg_names}")
                return None
            
            # 计算Lead I（可选）
            lead_i = None
            if self.process_lead_i:
                lead_i = self.lead_calculator.calculate_lead_i(ecg_data, ecg_names)
                if lead_i is None:
                    print(f"Warning: Cannot calculate Lead I for record {record['record_name']}")
                    return None
            
            # 计算Lead aVR（可选）
            lead_avr = None
            if self.process_lead_avr:
                lead_avr = self.lead_calculator.calculate_lead_avr(ecg_data, ecg_names)
                if lead_avr is None:
                    print(f"Warning: Cannot calculate Lead aVR for record {record['record_name']}")
                    return None
            
            # 获取PPG数据（通常第一个通道）
            ppg = ppg_data[0] 
            
            # 重采样
            lead_ii_resampled = self.signal_processor.resample_signal(
                lead_ii, ecg_fs, self.target_fs
            )
            ppg_resampled = self.signal_processor.resample_signal(
                ppg, ppg_fs, self.target_fs
            )
            
            # 如果需要处理Lead I，也进行重采样
            lead_i_resampled = None
            if self.process_lead_i and lead_i is not None:
                lead_i_resampled = self.signal_processor.resample_signal(
                    lead_i, ecg_fs, self.target_fs
                )
            
            # 如果需要处理Lead aVR，也进行重采样
            lead_avr_resampled = None
            if self.process_lead_avr and lead_avr is not None:
                lead_avr_resampled = self.signal_processor.resample_signal(
                    lead_avr, ecg_fs, self.target_fs
                )
            
            # 确定最小长度
            if lead_i_resampled is not None:
                min_length = min(len(lead_ii_resampled), len(ppg_resampled), 
                               len(lead_i_resampled))
            elif lead_avr_resampled is not None:
                min_length = min(len(lead_ii_resampled), len(ppg_resampled), 
                               len(lead_avr_resampled))
            else:
                min_length = min(len(lead_ii_resampled), len(ppg_resampled))
            
            # 切分成10秒片段
            processed_segments = []
            for i in range(0, min_length - self.segment_length + 1, self.segment_length):
                segment_data = {
                    'PPG': ppg_resampled[i:i+self.segment_length],
                    'ECG': lead_ii_resampled[i:i+self.segment_length]
                }
                
                # 如果处理了Lead I，添加到数据中
                if lead_i_resampled is not None:
                    segment_data['ECG_I'] = lead_i_resampled[i:i+self.segment_length]
                
                # 如果处理了Lead aVR，添加到数据中
                if lead_avr_resampled is not None:
                    segment_data['ECG_aVR'] = lead_avr_resampled[i:i+self.segment_length]
                
                processed_segments.append(segment_data)
            
            return processed_segments
            
        except Exception as e:
            record_name = record.get('record_name', 'unknown')
            print(f"\n❌ Error processing record: {record_name}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return None
    
    def read_arrow_file(self, arrow_file_path):
        """读取Arrow文件"""
        try:
            with pa.memory_map(str(arrow_file_path), "r") as source:
                reader = ipc.RecordBatchStreamReader(source)
                table = reader.read_all()
            df = table.to_pandas()
            return df
        except OSError as e:
            print(f"Skip corrupt Arrow file: {str(arrow_file_path)}")
            print(f"  Reason: {e}")
            return None
    
    def process_shard(self, shard_path):
        """
        处理单个shard
        
        Args:
            shard_path: shard目录路径
        """
        # 查找.arrow文件
        arrow_files = list(shard_path.glob('*.arrow'))
        
        if not arrow_files:
            print(f"No .arrow files found in {shard_path}")
            return
        
        for arrow_file in arrow_files:
            print(f"Processing {arrow_file}")
            
            # 读取记录
            records = self.read_arrow_file(arrow_file)
            if records is None:
                return
            
            for record in tqdm(records.iterrows(), desc=f"Processing {arrow_file.name}", 
                             total=len(records)):
                record = record[-1].to_dict()
                
                # 提取subject_id
                subject_id = self.extract_subject_id(record['record_name'])
                
                if subject_id is None:
                    continue
                
                # 处理记录
                processed_segments = self.process_record(record)
                
                if processed_segments is not None:
                    self.subject_data[subject_id].extend(processed_segments)
    
    def save_subject_data(self):
        """保存每个subject的所有10秒片段到npz文件"""
        print("\nSaving subject data...")
        
        for subject_id, segments in tqdm(self.subject_data.items()):
            if len(segments) == 0:
                continue
            
            # 准备保存的数据
            # 将所有片段堆叠成数组
            ppg_segments = [seg['PPG'] for seg in segments]
            ecg_segments = [seg['ECG'] for seg in segments]
            
            # 堆叠成 (N, L) 形状，N是片段数，L是每个片段的长度
            ppg_array = np.stack(ppg_segments, axis=0)  # shape: (N, 1250)
            ecg_array = np.stack(ecg_segments, axis=0)  # shape: (N, 1250)
            file_name = np.array([subject_id] * ppg_array.shape[0])
            
            # 准备保存字典
            save_dict = {
                'file_name': file_name,
                'PPG': ppg_array,
                'ECG': ecg_array
            }
            
            # 如果有Lead I数据，也添加
            if 'ECG_I' in segments[0]:
                ecg_i_segments = [seg['ECG_I'] for seg in segments]
                ecg_i_array = np.stack(ecg_i_segments, axis=0)
                save_dict['ECG_I'] = ecg_i_array
            if 'ECG_aVR' in segments[0]:
                ecg_avr_segments = [seg['ECG_aVR'] for seg in segments]
                ecg_avr_array = np.stack(ecg_avr_segments, axis=0)
                save_dict['ECG_aVR'] = ecg_avr_array
            
            # 保存为npz文件
            output_file = self.output_dir / f"{subject_id}.npz"
            np.savez(output_file, **save_dict)
            
            print(f"Saved {subject_id}: {len(segments)} segments, "
                  f"PPG shape: {ppg_array.shape}, ECG shape: {ecg_array.shape}")
    
    def run(self):
        """运行完整的pipeline"""
        print("Starting ECG/PPG Processing Pipeline (Modified)")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target sampling rate: {self.target_fs} Hz")
        print(f"Segment duration: {self.segment_duration} seconds")
        print(f"Process Lead I: {self.process_lead_i}")
        print("-" * 60)
        
        # 查找所有shard目录
        shard_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not shard_dirs:
            print("No shard directories found!")
            return
        
        print(f"Found {len(shard_dirs)} shard directories")
        
        # 处理每个shard
        for shard_dir in shard_dirs:
            print(f"\nProcessing shard: {shard_dir.name}")
            self.process_shard(shard_dir)
        
        # 保存结果
        self.save_subject_data()
        
        print("\n" + "=" * 60)
        print("Pipeline completed!")
        print(f"Processed {len(self.subject_data)} subjects")
        print(f"Output saved to: {self.output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ECG/PPG Data Processing Pipeline - Modified Version'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing shard folders with .arrow files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed .npz files'
    )
    parser.add_argument(
        '--target_fs',
        type=int,
        default=125,
        help='Target sampling rate in Hz (default: 125)'
    )
    parser.add_argument(
        '--segment_duration',
        type=int,
        default=10,
        help='Duration of each segment in seconds (default: 10)'
    )
    parser.add_argument(
        '--process_lead_i',
        action='store_true',
        help='Process and save Lead I data (default: False)'
    )
    parser.add_argument(
        '--process_lead_avr',
        action='store_true',
        help='Process and save Lead aVR data (default: False)'
    )
    
    args = parser.parse_args()
    
    # 创建并运行pipeline
    pipeline = ECGPPGPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_fs=args.target_fs,
        segment_duration=args.segment_duration,
        process_lead_i=args.process_lead_i
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()


'''
使用示例：

# 基本用法（只处理Lead II）
python ecg_ppg_pipeline.py \
    --input_dir /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecg \
    --output_dir /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed \
    --target_fs 125

# 同时处理Lead I和Lead II
python ecg_ppg_pipeline.py \
    --input_dir /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecg \
    --output_dir /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed \
    --target_fs 125 \
    --process_lead_i

python ecg_ppg_pipeline.py \
    --input_dir /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecg \
    --output_dir /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgAVR_ecgII-processed \
    --target_fs 125 \
    --process_lead_avr

# 自定义片段时长（例如5秒）
python ecg_ppg_pipeline.py \
    --input_dir /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecg \
    --output_dir /root/autodl-tmp/fengyuan/data/mimic-iv-processed \
    --target_fs 125 \
    --segment_duration 5 \
    --process_lead_i
'''