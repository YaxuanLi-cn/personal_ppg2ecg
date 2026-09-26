import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from skimage.metrics import structural_similarity
from dtw import dtw as dtw_metric
import os
import sys
from tqdm import tqdm
import torch
from pprint import pprint
import json

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

# 假设这些函数都在你的 evaluation.calculate_metric 中
from personal_ppg2ecg.v1.evaluation.calculate_metric import *

def calculate_fd_multi_sub(
    dir,
    gt_prefix,
    fake_prefix,
    pca_dim: Optional[int] = 64,
    eps: float = 1e-4,
    n_trials: int = 1
) -> Tuple[float, dict]:
    """遍历文件夹计算每个受试者的 FD"""
    fd_results = {}
    fd_mean_sum = 0
    
    # 获取两个目录下的文件并排序，确保一一对应
    gt_files = sorted([f for f in os.listdir(dir) if f.startswith(gt_prefix) and (f.endswith('.npz') or f.endswith('.npy'))])
    fake_files = sorted([f for f in os.listdir(dir) if f.startswith(fake_prefix) and (f.endswith('.npz') or f.endswith('.npy'))])
    
    count = 0
    for f_name, g_name in tqdm(zip(fake_files, gt_files), desc="Calculating Multi-Sub FID"):
        fake_path = os.path.join(dir, f_name)
        gt_path = os.path.join(dir, g_name)
        
        if not os.path.exists(gt_path):
            continue
            
        fake = np.load(fake_path)
        gt = np.load(gt_path)
        # if fake.size == 0 or gt.size == 0:
        #     # empty file
        #     continue
        n_samples_combined = len(fake) + len(gt)
        # 如果样本太少（比如小于 10），计算 FD 已经失去统计意义，建议跳过或设极小值
        if n_samples_combined <= 10:
            continue
        
        pca_dim = min(pca_dim, n_samples_combined)

        # 兼容 npz 格式
        if f_name.endswith('.npz'):
            fake = fake[fake.files[0]]
            gt = gt[gt.files[0]]

        fd_mean, fd_std = calculate_fd_for_small_sample(
            gt, fake, 
            pca_dim=pca_dim,
            eps=eps,
            n_trials=n_trials
        )
        
        fd_results[f_name] = {'mean': fd_mean, 'std': fd_std}
        fd_mean_sum += fd_mean
        count += 1
        
    avg_fd = fd_mean_sum / count if count > 0 else 0
    return avg_fd, fd_results

def calculate_FID_score_multi_sub(
    dir,
    gt_prefix,
    fake_prefix,
    model,
    batch_size=128, 
    device="cuda",
    eps=1e-6,
    pca_dim = None
) -> Tuple[float, dict]:
    """遍历文件夹计算每个受试者的 FID"""
    fid_results = {}
    fid_sum = 0
    
    gt_files = sorted([f for f in os.listdir(dir) if f.startswith(gt_prefix) and (f.endswith('.npz') or f.endswith('.npy'))])
    fake_files = sorted([f for f in os.listdir(dir) if f.startswith(fake_prefix) and (f.endswith('.npz') or f.endswith('.npy'))])
    
    count = 0
    for f_name, g_name in tqdm(zip(fake_files, gt_files), desc="Calculating Multi-Sub FID"):
        fake_path = os.path.join(dir, f_name)
        gt_path = os.path.join(dir, g_name)
        
        if not os.path.exists(gt_path):
            continue
            
        fake = np.load(fake_path)
        gt = np.load(gt_path)
        n_samples_combined = len(fake) + len(gt)
        # 如果样本太少（比如小于 10），计算 FD 已经失去统计意义，建议跳过或设极小值
        if n_samples_combined <= 10:
            continue
        
        if f_name.endswith('.npz'):
            fake = fake[fake.files[0]]
            gt = gt[gt.files[0]]

        # # 数据预处理与上采样
        # fake = resample_signals_scipy(fake, 125, 500)
        # gt = resample_signals_scipy(gt, 125, 500)
        
        # 提取特征
        fake_rep = compute_representations_in_batches(model, fake, batch_size=batch_size, device=device)
        real_rep = compute_representations_in_batches(model, gt, batch_size=batch_size, device=device)
        
        score = calculate_FID_score(fake_rep, real_rep, eps=eps, pca_dim=pca_dim)
        
        fid_results[f_name] = {'fid_score': score}
        fid_sum += score
        count += 1
        
    avg_fid = fid_sum / count if count > 0 else 0
    return avg_fid, fid_results


if __name__ == "__main__":
    # 配置路径
    base_path = "/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/rectified_flow_personal/mimic-iv-waveform/samples/"
    # base_path = "/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/rectified_flow_baseline/mimic-iv-waveform/samples/"
    # gt_all_path = os.path.join(base_path, "overall_gt_data.npy")
    # fake_all_path = os.path.join(base_path, "overall_fake_data.npy")
    
    # 填入你受试者级别的文件夹路径
    # multi_sub_dir = "/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/rectified_flow_personal/mimic-iv-waveform/samples_multi_sub"
    multi_sub_dir = "/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/rectified_flow_baseline/mimic-iv-waveform/samples_multi_sub" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ft_1lead_ECGFounder(device, use_bn=True, use_do=False)
    model.eval()

    # 5. 计算分受试者指标 (Multi-Subject)
    if os.path.exists(multi_sub_dir):
        avg_fd_sub, fd_sub_all = calculate_fd_multi_sub(
            dir=multi_sub_dir, 
            gt_prefix="overall_gt_data_",
            fake_prefix="overall_fake_data_"
            )
        avg_fid_sub, fid_sub_all = calculate_FID_score_multi_sub(
            dir=multi_sub_dir,
            gt_prefix="overall_gt_data_",
            fake_prefix="overall_fake_data_",
            model=model, 
            device=device)
        print(f"Avg Sub FD: {avg_fd_sub:.2f} | Avg Sub FID: {avg_fid_sub:.2f}")
        pprint(fd_sub_all)
        pprint(fid_sub_all)


        save_dir = os.path.join(multi_sub_dir, "stats")
        # 补全 FD 结果保存
        fd_save_path = os.path.join(save_dir, "fd_per_subject.json")
        with open(fd_save_path, 'w', encoding='utf-8') as f:
            # 使用 indent=4 让 JSON 文件人类可读
            json.dump(fd_sub_all, f, indent=4, ensure_ascii=False)
        print(f"Subject-wise FD scores saved to: {fd_save_path}")
        
        # 补全 FID 结果保存
        fid_save_path = os.path.join(save_dir, "fid_per_subject.json")
        with open(fid_save_path, 'w', encoding='utf-8') as f:
            json.dump(fid_sub_all, f, indent=4, ensure_ascii=False)
        print(f"Subject-wise FID scores saved to: {fid_save_path}")