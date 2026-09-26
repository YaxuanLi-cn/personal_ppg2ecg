import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os, sys, random
import numpy as np
from collections import defaultdict

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

# ---- 导入模型 ----
from personal_ppg2ecg.v0.model.Individual_base_extractor.ib_extractor import IBExtractor
from personal_ppg2ecg.v0.utils.ppgecg_dataset import PPGECGDataset


# ---- 配置 ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = "/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered/"
checkpoint_paths = {
    "Identity": None,
    "Untrained": None,
    "Iter-10000": "/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/IBExtractor/mimic-iv-waveform/checkpoints/IBExtractor-iter-10000.pth",
    "Iter-20000": "/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/IBExtractor/mimic-iv-waveform/checkpoints/IBExtractor-iter-20000.pth",
}
batch_size = 64
max_subjects = 10
samples_per_subject = 100


# ---- 加载数据 ----
dataset = PPGECGDataset(data_dir=data_path, train=False, return_type='rm')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---- 分层采样（仅执行一次） ----
# def stratified_sample_indices(subject_ids, max_subjects=10, samples_per_subject=100):
#     subj_to_indices = defaultdict(list)
#     for i, sid in enumerate(subject_ids):
#         subj_to_indices[int(sid.item())].append(i)

#     all_subjects = list(subj_to_indices.keys())
#     random.shuffle(all_subjects)
#     selected_subjects = all_subjects[:max_subjects]

#     selected_indices = []
#     for sid in selected_subjects:
#         indices = subj_to_indices[sid]
#         if len(indices) > samples_per_subject:
#             indices = random.sample(indices, samples_per_subject)
#         selected_indices.extend(indices)

#     return selected_indices, selected_subjects


# def stratified_sample_indices(subject_ids, max_subjects=10, samples_per_subject=50):
#     # 1. 建立 ID 到 索引 的映射
#     subj_to_indices = defaultdict(list)
#     for i, sid in enumerate(subject_ids):
#         subj_to_indices[int(sid.item())].append(i)

#     # 2. 核心修改：只保留样本量足够（>= samples_per_subject）的受试者
#     qualified_subjects = [
#         sid for sid, indices in subj_to_indices.items() 
#         if len(indices) >= samples_per_subject
#     ]
    
#     # 健壮性检查：如果满足条件的受试者不足 max_subjects
#     if len(qualified_subjects) < max_subjects:
#         print(f"Warning: Only {len(qualified_subjects)} subjects have >= {samples_per_subject} samples.")
#         selected_subjects = qualified_subjects
#     else:
#         # 3. 从合格的人中随机选出规定数量的受试者
#         selected_subjects = random.sample(qualified_subjects, max_subjects)

#     selected_indices = []
#     for sid in selected_subjects:
#         indices = subj_to_indices[sid]
#         # 4. 这里的 random.sample 不再需要 if 判断，因为上面过滤过了
#         sampled_indices = random.sample(indices, samples_per_subject)
#         selected_indices.extend(sampled_indices)

#     return selected_indices, selected_subjects


def stratified_sample_indices(subject_ids, max_subjects=10, samples_per_subject=100):
    # 1. 建立 ID 到 索引 的映射
    subj_to_indices = defaultdict(list)
    for i, sid in enumerate(subject_ids):
        subj_to_indices[int(sid.item())].append(i)

    # 2. 将受试者分类
    qualified_subjects = [sid for sid, idxs in subj_to_indices.items() if len(idxs) >= samples_per_subject]
    unqualified_subjects = [sid for sid, idxs in subj_to_indices.items() if len(idxs) < samples_per_subject]
    
    # 3. 达标组保持随机性，确保每次看到的簇不同
    random.shuffle(qualified_subjects)

    # 4. 核心修改：未达标组按样本量降序排列，优先选“大户”
    # 使用 lambda 检查 len(subj_to_indices[sid])
    unqualified_subjects.sort(key=lambda sid: len(subj_to_indices[sid]), reverse=True)

    # 5. 混合选择
    if len(qualified_subjects) >= max_subjects:
        selected_subjects = qualified_subjects[:max_subjects]
    else:
        # 达标的不够，从未达标组中选出样本量最多的前 N 个补齐
        needed = max_subjects - len(qualified_subjects)
        selected_subjects = qualified_subjects + unqualified_subjects[:needed]
        print(f"Sampling Info: Using {len(qualified_subjects)} qualified and "
              f"{len(selected_subjects) - len(qualified_subjects)} best-available subjects.")

    # 6. 提取索引
    selected_indices = []
    for sid in selected_subjects:
        indices = subj_to_indices[sid]
        count = min(len(indices), samples_per_subject)
        # 随机采样以保证代表性
        sampled_indices = random.sample(indices, count)
        selected_indices.extend(sampled_indices)

    return selected_indices, selected_subjects


# ---- 第一步：固定采样的数据 ----
all_ppg, all_ecg, all_subj = [], [], []
for ppg, ecg, subject_id in dataloader:
    all_ppg.append(ppg)
    all_ecg.append(ecg)
    all_subj.append(subject_id)
all_ppg = torch.cat(all_ppg, dim=0)
all_ecg = torch.cat(all_ecg, dim=0)
all_subj = torch.cat(all_subj, dim=0)

selected_indices, selected_subjects = stratified_sample_indices(all_subj, max_subjects, samples_per_subject)
print(f"✅ Selected {len(selected_subjects)} subjects, total {len(selected_indices)} samples")

# 取出固定的子集
ppg_subset = all_ppg[selected_indices].unsqueeze(-1).to(device).float()
ecg_subset = all_ecg[selected_indices].unsqueeze(-1).to(device).float()
subj_subset = all_subj[selected_indices].numpy()


# ---- 通用的特征提取函数 ----
def extract_features(model, ppg, ecg, identity=False):
    if identity:
        concat_raw = torch.cat([ppg.squeeze(-1), ecg.squeeze(-1)], dim=-1)
        features = torch.nn.functional.adaptive_avg_pool1d(concat_raw.unsqueeze(1), 256).squeeze(1)
        features = F.normalize(features, dim=-1)
    else:
        model.eval()
        with torch.no_grad():
            features = model(ppg, ecg, reduce=True)
            features = F.normalize(features, dim=-1)
    return features.cpu().numpy()


# ---- 主循环：对相同数据提取四种embedding ----
results = {}
for name, ckpt_path in checkpoint_paths.items():
    print(f"\n>>> Processing {name} ...")
    if ckpt_path is None and name != "Identity":
        model = IBExtractor().to(device)  # 未训练模型
    elif ckpt_path is not None:
        model = IBExtractor().to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        model = None  # identity模式不需要模型

    embeddings = extract_features(model, ppg_subset, ecg_subset, identity=(name == "Identity"))
    results[name] = (embeddings, subj_subset)


# ---- 统一颜色映射 ----
unique_subjects = np.unique(subj_subset)
palette = sns.color_palette("hls", len(unique_subjects))
subject_to_color = {sid: palette[i % len(palette)] for i, sid in enumerate(unique_subjects)}


# ---- t-SNE 可视化 ----
plt.figure(figsize=(18, 16))
names = list(results.keys())

for i, name in enumerate(names):
    embeddings, subjects = results[name]
    print(f">>> Running t-SNE for {name} ({embeddings.shape[0]} samples)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_tsne = tsne.fit_transform(embeddings)

    colors = [subject_to_color[sid] for sid in subjects]
    plt.subplot(2, 2, i + 1)
    plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=colors, s=40)
    plt.title(name)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

# ---- 统一图例 ----
handles = [plt.Line2D([0], [0], marker='o', color=c, linestyle='', markersize=8) 
           for c in palette[:len(unique_subjects)]]
labels = [str(sid) for sid in unique_subjects]
plt.legend(handles, labels, title="Subject ID", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/drawer/tSNE_IBExtractor_Identity_vs_Training.png")
plt.show()
