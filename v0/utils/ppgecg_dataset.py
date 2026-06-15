import os
import json
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

# class PPGECGDataset(Dataset):
#     '''
#     - PPG, lead II
#     - support return types:
#         1. PPG, ECG(lead II)
#         2. PPG-ref, ECG-ref(lead II), PPG, ECG(lead II)
#         3. PPG-ref, ECG-ref(lead II)
#     - support pre- z-score normalization for PPG and ECG(lead II)
#     '''

#     def __init__(self, data_dir, train=True, return_type='pf'):
#         self.data_dir = os.path.join(data_dir, "train" if train else "test")
#         self.record_to_file_map_path = os.path.join(self.data_dir, "record_to_file_map.json")
#         self.files = sorted(glob(os.path.join(self.data_dir, "*.npz")))
#         self.record_to_file_map = self.get_record_to_file_map(self.record_to_file_map_path)
#         self.curr_file = None
#         self.curr_data = None 
#         self.z_score_path = os.path.join(data_dir, "z_score_mean_std.json")
#         with open(self.z_score_path, "r") as f:
#             self.z_score = json.load(f)['train']
#         self.return_type = return_type

#     def get_record_to_file_map(self, record_to_file_map=None):
#         if record_to_file_map and os.path.exists(record_to_file_map):
#             with open(record_to_file_map, "r") as f:
#                 record_to_file = json.load(f)
#                 record_to_file = {int(k): v for k, v in record_to_file.items()}
#             return record_to_file
#         else:
#             record_to_file = {}
#             for file in self.files:
#                 data_in_file = np.load(file, allow_pickle=True)
#                 file_name = data_in_file["file_name"]
#                 num_data_in_file = len(file_name) ####
#                 cur_len = len(record_to_file)
#                 for i in range(num_data_in_file):
#                     record_to_file[cur_len + i] = {
#                         'file_path': file,
#                         'idx_in_file': i
#                     }
#             if record_to_file_map:
#                 with open(record_to_file_map, "w") as f:
#                     json.dump(record_to_file, f, indent=2)
#             return record_to_file
        
#     def __len__(self):
#         return len(self.record_to_file_map.keys())
    
#     def cache_load(self, file):
#         if self.curr_file != file:
#             self.curr_data = np.load(file, allow_pickle=True)
#             self.curr_file = file
#         return self.curr_data

#     def idx_to_item(self, data, idx, norm=True):
#         # extract PPG-ECG pair data from data_in_file based on index, support z-score normalization
#         ppg = data["PPG"][idx] 
#         ecg = data["ECG"][idx]
#         if norm:
#             ppg = (ppg - self.z_score['PPG']['mean']) / self.z_score['PPG']['std']
#             ecg = (ecg - self.z_score['ECG']['mean']) / self.z_score['ECG']['std']
#         ppg = torch.FloatTensor(ppg)
#         ecg = torch.FloatTensor(ecg)

#         return ppg, ecg

#     def __getitem__(self, idx):
#         record_info = self.record_to_file_map[idx]
#         file, idx_in_file = record_info['file_path'], record_info['idx_in_file']
#         data_in_file = self.cache_load(file)
#         # data_in_file = np.load(file, mmap_mode="r")
#         if self.return_type == 'pf':
#             # PPGFlowECG
#             ppg, ecg = self.idx_to_item(data_in_file, idx_in_file, norm=True)
#             return ppg, ecg
        
#         elif self.return_type == 'ppf':
#             # personalized PPGFlowECG
#             ppg, ecg = self.idx_to_item(data_in_file, idx_in_file, norm=True)
#             num_data_in_file = len(data_in_file["file_name"])
#             rand_idx = random.choice([i for i in range(num_data_in_file) if i != idx_in_file])
#             ppg_ref, ecg_ref = self.idx_to_item(data_in_file, rand_idx, norm=True)
#             return ppg, ecg, ppg_ref, ecg_ref
        
#         elif self.return_type == 'rm':
#             # representation model
#             ppg, ecg = self.idx_to_item(data_in_file, idx_in_file, norm=True)
#             subject_id = data_in_file["file_name"][idx_in_file]
#             subject_id = torch.tensor(int(subject_id[1:]), dtype=torch.int64)
#             return ppg, ecg, subject_id

#         else:
#             raise NotImplementedError(f"Unsupported return type: {self.return_type}")

# class PPGECGDatasetv2(Dataset):
#     '''
#     PPG, lead I, lead II
#     support return types:
#     1. PPG, ECG(lead II)
#     2. PPG-ref, ECG-ref(lead I)
#     3. PPG-ref, ECG-ref(lead I), PPG, ECG(lead II)
#     '''

#     def __init__(self, data_dir, train=True):
#         self.data_dir = os.path.join(data_dir, "train" if train else "test")
#         self.record_to_file_map_path = os.path.join(self.data_dir, "train" if train else "test", "record_to_file_map.json")
#         self.files = sorted(glob(os.path.join(self.data_dir, "*.npz")))
#         self.record_to_file_map = self.get_record_to_file_map(self.record_to_file_map_path)
#         self.curr_file = None
#         self.curr_data = None 

#     def get_record_to_file_map(self, record_to_file_map=None):
#         if record_to_file_map and os.path.exists(record_to_file_map):
#             with open(record_to_file_map, "r") as f:
#                 record_to_file = json.load(f)
#                 record_to_file = {int(k): v for k, v in record_to_file.items()}
#             return record_to_file
#         record_to_file = {}
#         for file in self.files:
#             data_in_file = np.load(file, allow_pickle=True)
#             file_name = data_in_file["file_name"]
#             num_data_in_file = len(file_name) ####
#             cur_len = len(record_to_file)
#             for i in range(num_data_in_file):
#                 record_to_file[cur_len + i] = {
#                     'file_path': file,
#                     'idx_in_file': i
#                 }
#         if record_to_file_map:
#             with open(record_to_file_map, "w") as f:
#                 json.dump(record_to_file, f, indent=2)
#         return record_to_file
        
#     def __len__(self):
#         return len(self.record_to_file_map.keys())
    
#     def cache_load(self, file):
#         if self.curr_file != file:
#             self.curr_data = np.load(file, allow_pickle=True)
#             self.curr_file = file
#         return self.curr_data


#     def __getitem__(self, idx, return_type='pf'):
#         record_info = self.record_to_file_map[idx]
#         file, idx_in_file = record_info['file_path'], record_info['idx_in_file']
#         data_in_file = self.cache_load(file)
#         if return_type == 'pf':
#             # PPGFlowECG
#             ppg = data_in_file["PPG"][idx_in_file]
#             ecg = data_in_file["ECG"][idx_in_file]
#             ppg = torch.FloatTensor(ppg)
#             ecg = torch.FloatTensor(ecg)
#             return ppg, ecg
#         elif return_type == 'ppf':
#             # personalized PPGFlowECG
#             ppg = data_in_file["PPG"][idx_in_file]
#             ecg = data_in_file["ECG"][idx_in_file]
#             num_data_in_file = len(data_in_file["file_name"])
#             rand_idx = random.choice([i for i in range(num_data_in_file) if i != idx_in_file])
#             ppg_ref = data_in_file["PPG"][rand_idx]
#             ecg_ref = data_in_file["ECG_I"][rand_idx]
#             ppg = torch.FloatTensor(ppg)
#             ecg = torch.FloatTensor(ecg)
#             ppg_ref = torch.FloatTensor(ppg_ref)
#             ecg_ref = torch.FloatTensor(ecg_ref)
#             return ppg, ecg, ppg_ref, ecg_ref
#         elif return_type == 'rm':
#             # representation model
#             ppg = data_in_file["PPG"][idx_in_file]
#             ecg_i = data_in_file["ECG_I"][idx_in_file]
#             ppg = torch.FloatTensor(ppg)
#             ecg_i = torch.FloatTensor(ecg_i)
#             return ppg, ecg_i
#         else:
#             raise ValueError(f"Unsupported return type: {return_type}")


class PPGECGDataset(Dataset):
    """
    支持：
      - return_type='pf': 返回 (ppg, ecg)
      - return_type='rm': 返回 (ppg, ecg, subject_id)
    """

    def __init__(self, data_dir, train=True, return_type='pf'):
        if data_dir.endswith('.pt'):
            self.data_path = data_dir
        else:
            self.data_path = os.path.join(data_dir, "train.pt" if train else "test.pt")
        self.z_score_path = os.path.join(data_dir, "z_score_mean_std.json")

        # 加载数据（完全在CPU上）
        data = torch.load(self.data_path, map_location='cpu')
        self.ppg = data['PPG']
        self.ecg = data['ECG']
        self.file_name = data['file_name']

        with open(self.z_score_path, "r") as f:
            self.z_score = json.load(f)['train']

        self.return_type = return_type

    def __len__(self):
        return len(self.ppg)

    def idx_to_item(self, idx, norm=True):
        # (L,) float32 tensors
        subject_id = self.file_name[idx].clone()
        ppg = self.ppg[idx].clone()
        ecg = self.ecg[idx].clone()

        if norm:
            ppg = (ppg - self.z_score['PPG']['mean']) / (self.z_score['PPG']['std'] + 1e-8)
            ecg = (ecg - self.z_score['ECG']['mean']) / (self.z_score['ECG']['std'] + 1e-8)

        return ppg, ecg, subject_id

    def __getitem__(self, idx):
        # PPGFlowECG 模式
        ppg, ecg, subject_id = self.idx_to_item(idx)
        if self.return_type == 'pf':
            return ppg, ecg
        elif self.return_type == 'rm':
            return ppg, ecg, subject_id
        else:
            raise NotImplementedError(f"Unsupported return type: {self.return_type}")


class PPGECGPairDataset(Dataset):
    """
    支持：
      - return_type='pf': 返回 (ppg, ecg)
      - return_type='ppf': 返回 (ppg, ecg, ppg-ref, ecg-ref)
    """
    def __init__(self, data_dir, train=True, return_type='pf'):
        if data_dir.endswith('.pt'):
            print(data_dir)
            self.data_path = data_dir
            self.z_score_path = os.path.join(os.path.dirname(os.path.dirname(data_dir)), "z_score_mean_std.json")
        else:
            self.data_path = os.path.join(data_dir, "train.pt" if train else "test.pt")
            self.z_score_path = os.path.join(data_dir, "z_score_mean_std.json")

        # 加载数据（完全在CPU上）
        data = torch.load(self.data_path, map_location='cpu')
        self.ppg = data['PPG']
        self.ecg = data['ECG']
        self.file_name = data['file_name']

        self.subject_id_to_idx_map = self.get_subject_id_to_idx_map()

        with open(self.z_score_path, "r") as f:
            self.z_score = json.load(f)['train']
        
        self.return_type = return_type

    def __len__(self):
        return len(self.ppg)
    
    def get_subject_id_to_idx_map(self):
        subject_id_to_idx_map = {}
        for i in range(len(self.ppg)):
            subject_id = self.file_name[i]
            if int(subject_id) not in subject_id_to_idx_map.keys():
                subject_id_to_idx_map[int(subject_id)] = [i]
            else:
                subject_id_to_idx_map[int(subject_id)].append(i)
        return subject_id_to_idx_map


    def idx_to_item(self, idx, norm=True):
        subject_id = self.file_name[idx].clone()
        ppg = self.ppg[idx].clone()
        ecg = self.ecg[idx].clone()

        if norm:
            ppg = (ppg - self.z_score['PPG']['mean']) / (self.z_score['PPG']['std'] + 1e-8)
            ecg = (ecg - self.z_score['ECG']['mean']) / (self.z_score['ECG']['std'] + 1e-8)

        return ppg, ecg, subject_id

    def __getitem__(self, idx):
        # PPGFlowECG 模式
        ppg, ecg, subject_id = self.idx_to_item(idx)
        if self.return_type == 'pf':
            return ppg, ecg
        elif self.return_type == 'ppf':
            ref_idx = random.choice([i for i in self.subject_id_to_idx_map[int(subject_id)] if i != idx])
            ppg_ref, ecg_ref, subject_id_ = self.idx_to_item(ref_idx)
            return ppg, ecg, ppg_ref, ecg_ref




if __name__ == "__main__":
    # dataset = PPGECGDataset(
    #     # data_dir="/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed-filtered",
    #     # record_to_file_map="/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed-filtered/record_to_file_map.json"
    #     data_dir="/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered",
    #     train=True,
    #     return_type='pf'
    #     )
    # print(f"Dataset length: {len(dataset)}")


    # 设置随机种子保证复现性
    random.seed(42)
    torch.manual_seed(42)

    # 初始化数据集
    data_dir = "/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered"
    dataset = PPGECGPairDataset(data_dir=data_dir, train=True)

    # 随机抽 100 条样本
    num_samples = 100
    print(f"✅ Sampling {num_samples} pairs...\n")

    same_subject_count = 0

    for _ in range(num_samples):
        ppg, ecg, subject_id, ppg_ref, ecg_ref, subject_id_ = dataset[random.randint(0, len(dataset)-1)]
        print(subject_id, subject_id_)
        if subject_id == subject_id_:
            same_subject_count += 1

    print(f"✅ 共测试 {num_samples} 对样本，其中 {same_subject_count} 对 subject_id 一致。")
    print(f"✅ 一致比例：{same_subject_count / num_samples:.2%}")

    # import time
    # # 测试读取前 10 条数据
    # idxs = np.random.choice(len(dataset), 100, replace=False)
    # start_time = time.time()
    # for i in idxs:
    #     ppg, ecg = dataset[i]  # 默认 return_type='pf'
    #     print(f"Item {i} - PPG shape: {ppg.shape}, ECG shape: {ecg.shape}, subject_id: {None}")
    # end_time = time.time()
    # print(f"Time to read 10 items: {end_time - start_time:.4f} s")

    # # 测试随机读取 100 条数据
    # start_time = time.time()
    # indices = np.random.choice(len(dataset), 100, replace=False)
    # for idx in indices:
    #     ppg, ecg = dataset[idx]
    # end_time = time.time()
    # print(f"Time to read 100 random items: {end_time - start_time:.4f} s")
    # import time
    # from torch.utils.data import DataLoader

    # loader = DataLoader(
    #     dataset,
    #     batch_size=32,      # 每个 batch 32 条
    #     shuffle=True,       # 随机打乱
    #     num_workers=4,      # 多进程并行读取
    #     pin_memory=True     # 加速 GPU 训练
    # )

    # # 测试读取前几个 batch
    # n_items = 100
    # n_batches = (n_items + 31) // 32  # batch_size=32
    # start_time = time.time()
    # count = 0
    # for batch in loader:
    #     ppg_batch, ecg_batch = batch  # 每个 batch shape: (batch_size, 1250)
    #     for i in range(ppg_batch.shape[0]):
    #         print(f"Item {count} - PPG shape: {ppg_batch[i].shape}, ECG shape: {ecg_batch[i].shape}")
    #     count += 1
    #     if count >= n_batches:
    #         break
    #     #     count += 1
    #     #     if count >= n_items:
    #     #         break
    #     # if count >= n_items:
    #     #     break
    # end_time = time.time()
    # print(f"Time to read {n_items} items via DataLoader: {end_time - start_time:.4f} s")
