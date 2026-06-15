# personal_ppg2ecg 使用文档

`personal_ppg2ecg` 是一个基于 PPG 生成 ECG 的研究项目，包含了数据下载、预处理、个人化表示学习、latent rectified flow 建模、结果评估和可视化等完整流程。

项目整体上可以理解为三层：

1. 原始数据处理：把原始波形整理成训练可用的 `.npz` / `.pt` 数据。
2. 表示学习：训练 `IBExtractor` 和 `CardioAlign VAE`，为后续生成模型提供个体表示和潜空间。
3. 生成建模：训练 `RectifiedFlow` / `PersonalRectifiedFlow`，实现从 PPG 到 ECG 的条件生成。

---

## 目录说明

### `data_pipeline/`
原始数据获取和预处理脚本。

- `downloads.py`：从 Hugging Face 数据集下载 `.hea` 文件。
- `download.py` / `download2.py`：其他下载脚本。
- `ecg_ppg_pipeline.py`：PPG/ECG 原始数据清洗、切窗、重采样、质量控制。
- `pl_test.py`：测试脚本。

### `v0/`
主实验代码。

- `main.py`：生成模型训练/采样主入口。
- `engine/`：训练器、学习率调度、日志。
- `model/`
  - `Individual_base_extractor/`：个体表示提取器 `IBExtractor`
  - `cardioalign_encoder/`：PPG/ECG 对齐 VAE 编码器
  - `latent_rectified_flow/`：Rectified Flow / Personal Rectified Flow
- `utils/`：数据集、配置和工具函数。
- `evaluation/`：FD/FID 等评估指标。
- `drawer/`：可视化脚本，例如 t-SNE。
- `config/`：各模型配置文件。

---

## 环境安装

项目的依赖可以参考：

- `personal_ppg2ecg/v0/requirements.txt`
- `personal_ppg2ecg/v0/environment.yml`

### 推荐安装方式

```bash
cd personal_ppg2ecg/v0
pip install -r requirements.txt
```

如果你使用 Conda，也可以：

```bash
conda env create -f environment.yml
conda activate <env_name>
```

### 关键依赖

项目中常见的核心依赖包括：

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `tqdm`
- `huggingface-hub`
- `neurokit2`
- `biosppy`
- `biobss`
- `wfdb`

---

## 数据说明

项目默认使用 MIMIC-IV Waveform 相关数据，并假定已经整理成项目读取的格式。

### 预处理后的数据格式

`v0/utils/ppgecg_dataset.py` 中的数据集类默认读取：

- `train.pt`
- `test.pt`
- `z_score_mean_std.json`

其中 `.pt` 文件通常包含：

- `PPG`
- `ECG`
- `file_name`

如果是个人化实验，还会使用 `subject_id` 或 `file_name` 中的受试者标识。

### 路径约定

代码里存在一些硬编码路径，常见如下：

- 原始/处理后数据：
  - `/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered`
- VAE checkpoint：
  - `/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/cardioalign_encoder/...`
- IBExtractor checkpoint：
  - `/root/autodl-tmp/fengyuan/projects/person_ppg2ecg/v0/results/IBExtractor/...`

如果你在本地或其他服务器运行，需要把这些路径改成你的实际目录。

---

## 数据下载

`personal_ppg2ecg/data_pipeline/downloads.py` 使用 Hugging Face Hub 下载数据集 `lucky9-cyou/MIMIC-IV-Waveform`。

### 功能

- 自动重试
- 断点续传
- 只下载 `.hea` 文件

### 使用方法

```bash
python personal_ppg2ecg/data_pipeline/downloads.py
```

### 说明

脚本中默认只下载：

```python
ALLOW_PATTERNS = "**/*.hea"
```

如果你需要下载更多文件类型，可以修改 `ALLOW_PATTERNS`。

---

## 数据预处理流程

项目的预处理核心在 `ecg_ppg_pipeline.py`，整体思路是：

1. 读取原始 `.npz`
2. 对 PPG 和 ECG 做平滑
3. 做质量控制
4. 保存过滤后的结果

`step1.py`、`step2.py`、`step2_fast.py` 是两阶段/加速版处理脚本，通常可按下面逻辑理解：

- `step1.py`：初步切窗、对齐、导出初始数据
- `step2.py`：平滑 + 质量过滤
- `step2_fast.py`：使用多进程加速的 `step2`

### 典型输入输出

- 输入：原始或中间结果 `.npz`
- 输出：过滤后 `.npz`，包含
  - `PPG`
  - `ECG`
  - `ECG_I`（如存在）
  - `file_name`

### 运行示例

```bash
python personal_ppg2ecg/v0/data_process_to_npz/step2.py \
  --input /path/to/mimic-iv-aligned-ppg_ecgII-processed \
  --output /path/to/mimic-iv-aligned-ppg_ecgII-processed-filtered
```

如果你想更快处理大批量数据，可以用：

```bash
python personal_ppg2ecg/v0/data_process_to_npz/step2_fast.py \
  --input /path/to/mimic-iv-aligned-ppg_ecgII-processed \
  --output /path/to/mimic-iv-aligned-ppg_ecgII-processed-filtered
```

---

## 数据读取方式

`v0/utils/ppgecg_dataset.py` 提供了两个主要数据集：

### `PPGECGDataset`

用于：

- `return_type='pf'`：返回 `(ppg, ecg)`
- `return_type='rm'`：返回 `(ppg, ecg, subject_id)`

读取格式：

- `data_dir/train.pt` 或 `data_dir/test.pt`
- `data_dir/z_score_mean_std.json`

### `PPGECGPairDataset`

用于：

- `return_type='pf'`：返回 `(ppg, ecg)`
- `return_type='ppf'`：返回 `(ppg, ecg, ppg_ref, ecg_ref)`

这个类被 `main.py` 中的生成模型训练和采样流程使用得更多。

---

## 模型结构

### 1. `IBExtractor`

位置：

- `v0/model/Individual_base_extractor/ib_extractor.py`
- 训练脚本：`v0/model/Individual_base_extractor/train.py`

作用：

- 从 `(PPG, ECG)` 对中学习个体相关表征
- 通过对比学习，让同一个受试者的样本更接近

输出用途：

- 为 personal rectified flow 提供个体嵌入

### 2. `CardioAlign VAE`

位置：

- `v0/model/cardioalign_encoder/cardioalign_model.py`
- 训练脚本：`v0/model/cardioalign_encoder/train.py`

作用：

- 对 PPG 和 ECG 进行潜空间编码
- 学习跨模态对齐的 latent space

训练时还可以加入：

- latent alignment loss
- cross reconstruction loss
- InfoNCE loss

### 3. `Latent Rectified Flow`

位置：

- `v0/model/latent_rectified_flow/rectified_flow.py`
- `v0/model/latent_rectified_flow/transformer.py`

作用：

- 在 latent space 中学习条件生成
- 输入 PPG latent，输出 ECG latent 或 ECG 相关目标

配置文件：

- `v0/config/latent_rectified_flow.yaml`
- `v0/config/personal_latent_rectified_flow.yaml`

两者差别主要是：

- `latent_rectified_flow.yaml`：baseline
- `personal_latent_rectified_flow.yaml`：带个人化 embedding

---

## 训练流程

建议训练顺序：

1. 数据预处理
2. 训练 `CardioAlign VAE`
3. 训练 `IBExtractor`
4. 训练 baseline 或 personal rectified flow
5. 做采样和评估

---

## 1. 训练 IBExtractor

训练脚本：

```bash
python personal_ppg2ecg/v0/model/Individual_base_extractor/train.py \
  --config personal_ppg2ecg/v0/config/ib_extractor.yaml \
  --save_dir personal_ppg2ecg/v0/results/IBExtractor
```

### 配置文件

`v0/config/ib_extractor.yaml` 里主要包括：

- `train.lr`
- `train.batch_size`
- `train.total_iterations`
- `train.save_interval`
- `model.embed_dim`
- `model.in_channel`
- `model.num_layers`

### 输出

checkpoint 会保存到：

```text
results/IBExtractor/mimic-iv-waveform/checkpoints/
```

文件名类似：

```text
IBExtractor-iter-10000.pth
```

---

## 2. 训练 CardioAlign VAE

训练脚本：

```bash
python personal_ppg2ecg/v0/model/cardioalign_encoder/train.py \
  --config personal_ppg2ecg/v0/config/cardioalign_encoder.yaml \
  --save_dir personal_ppg2ecg/v0/results/cardioalign_encoder
```

### 配置文件

`v0/config/cardioalign_encoder.yaml` 里常见参数：

- `train.lr`
- `train.batch_size`
- `train.total_iterations`
- `train.kld_weight`
- `train.lambda_align`
- `train.lambda_cross`
- `train.lambda_infonce`
- `model.decoder.seq_len`
- `model.decoder.latent_dim`

### 输出

checkpoint 会保存到：

```text
results/cardioalign_encoder/mimic-iv-waveform/checkpoints/
```

文件名类似：

```text
VAE-iter-20000.pth
```

---

## 3. 训练 Rectified Flow / Personal Rectified Flow

主入口：

```bash
python personal_ppg2ecg/v0/main.py \
  --train \
  --config_file personal_ppg2ecg/v0/config/latent_rectified_flow.yaml \
  --model_type pf \
  --name latent_rectified_flow \
  --output baseline
```

如果是 personal 版本：

```bash
python personal_ppg2ecg/v0/main.py \
  --train \
  --config_file personal_ppg2ecg/v0/config/personal_latent_rectified_flow.yaml \
  --model_type ppf \
  --name personal_latent_rectified_flow \
  --output personal
```

### 参数说明

- `--train`：开启训练
- `--config_file`：配置文件
- `--model_type`
  - `pf`：baseline，输入 PPG -> ECG
  - `ppf`：personalized，输入 PPG + reference pair
- `--name`：实验名
- `--output`：结果目录名

### `main.py` 的数据读取

代码中默认读取：

```python
PPGECGPairDataset(
    data_dir="/root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered",
    train=True/False,
    return_type=args.model_type
)
```

也就是说，训练前需要先把数据预处理好，并放到这个目录结构下。

### 生成模型输出

`Trainer` 会把 checkpoint 保存到：

```text
results/rectified_flow_baseline/mimic-iv-waveform/checkpoints/
results/rectified_flow_personal/mimic-iv-waveform/checkpoints/
```

文件名类似：

```text
checkpoint-1.pt
checkpoint-2.pt
```

---

## 采样与推理

`main.py` 在 `--train` 关闭时会进入采样逻辑。

### baseline 采样示例

```bash
python personal_ppg2ecg/v0/main.py \
  --config_file personal_ppg2ecg/v0/config/latent_rectified_flow.yaml \
  --model_type pf \
  --mode synthesis \
  --milestone 1 \
  --name latent_rectified_flow
```

### personal 采样示例

```bash
python personal_ppg2ecg/v0/main.py \
  --config_file personal_ppg2ecg/v0/config/personal_latent_rectified_flow.yaml \
  --model_type ppf \
  --mode synthesis \
  --milestone 1 \
  --name personal_latent_rectified_flow
```

### 采样参数

- `--milestone`：加载第几个 checkpoint，对应 `checkpoint-{milestone}.pt`
- `sample.sampling_steps`：采样步数
- `sample.subset_save_threshold`：保存子集阈值

### 采样结果

采样结果会保存在：

```text
results/.../samples/
```

具体保存形式由 `Trainer.sample_shift(...)` 决定，通常包括生成信号及相关中间文件。

---

## 评估

评估脚本：

```bash
python personal_ppg2ecg/v0/evaluation/evaluate.py
```

### 主要指标

- `FD`：Fréchet Distance
- `FID`：使用特征提取器后的 Fréchet 类指标

脚本会对每个受试者分别计算指标，然后统计平均值，并保存：

- `fd_per_subject.json`
- `fid_per_subject.json`

### 说明

`evaluate.py` 中默认路径是硬编码的，运行前需要根据你的实际结果目录修改：

```python
base_path = ".../results/rectified_flow_personal/mimic-iv-waveform/samples/"
multi_sub_dir = ".../results/rectified_flow_baseline/mimic-iv-waveform/samples_multi_sub"
```

另外，`evaluate.py` 中使用了 `ft_1lead_ECGFounder` 之类的特征模型名，若你本地缺少对应实现，需要补齐或替换。

---

## 可视化

### t-SNE 示例

脚本：

```bash
python personal_ppg2ecg/v0/drawer/ibe_t_sne.py
```

它会：

1. 读取固定的 PPG/ECG 数据子集
2. 使用 `IBExtractor` 的不同 checkpoint 提取特征
3. 对特征做 t-SNE
4. 按受试者 ID 着色并保存图像

默认输出：

```text
v0/results/drawer/tSNE_IBExtractor_Identity_vs_Training.png
```

### 需要注意

脚本内部有固定路径，例如：

- `data_path`
- `checkpoint_paths`

运行前要改成你机器上的实际路径。

---

## 配置文件速览

### `v0/config/latent_rectified_flow.yaml`

baseline 生成模型配置，核心字段：

- `model.target`
- `model.params.feature_size`
- `model.params.temporal_size`
- `solver.results_folder`
- `solver.vae.checkpoint`
- `solver.ibe.checkpoint`
- `data.return_type: pf`

### `v0/config/personal_latent_rectified_flow.yaml`

personal 版本配置，核心字段：

- `model.target`
- `model.params.if_personal: True`
- `model.params.personal_emb_dim`
- `data.return_type: ppf`

### `v0/config/cardioalign_encoder.yaml`

VAE 训练配置，核心字段：

- `train.batch_size`
- `train.total_iterations`
- `train.lambda_align`
- `train.lambda_cross`
- `train.lambda_infonce`

### `v0/config/ib_extractor.yaml`

IBExtractor 配置，核心字段：

- `train.batch_size`
- `train.total_iterations`
- `model.embed_dim`
- `model.in_channel`
- `model.num_layers`

---

## 实际使用建议

1. 先确认数据格式正确。
2. 先跑预处理，生成过滤后的 `.npz` / `.pt`。
3. 先训练 `CardioAlign VAE` 和 `IBExtractor`，再训练 `RectifiedFlow`。
4. 如果你只是想快速验证生成流程，可以先用 `latent_rectified_flow.yaml` 跑 baseline。
5. 如果你要做个体化建模，再切到 `personal_latent_rectified_flow.yaml`。

---

## 常见修改点

如果你要在自己的环境里跑，最常改的地方是：

- `data_pipeline/downloads.py` 中的 `LOCAL_DIR`
- `main.py` 中的数据路径
- `main.py` / `train.py` 中的 checkpoint 路径
- 各个 `config/*.yaml` 里的 `results_folder`

---

## 备注

- 这个项目的很多路径是为作者的实验环境写死的，迁移时需要手动替换。
- `main.py` 会将 `TRANSFORMERS_OFFLINE` 设为 `1`，适合离线环境。
- 代码里部分脚本偏研究型，运行前建议先确认本地是否已经有对应 checkpoint 和数据文件。

