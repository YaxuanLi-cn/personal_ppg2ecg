#!/usr/bin/env python3
import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError, HfHubHTTPError

# =============================
# 基本配置
# =============================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像（可选）

REPO_ID = "lucky9-cyou/mimic-iv-aligned-ppg-ecg"
LOCAL_DIR = "/root/autodl-tmp/personal_ppg2ecg/mimic-iv-aligned-ppg-ecg"
TOKEN = ""  # 替换为你的 HuggingFace Token
ALLOW_PATTERNS = None  # 下载全部内容，如果只想下载部分可改成 "subfolder/*"

# 重试设置
MAX_RETRIES = 9999     # 无限重试
RETRY_INTERVAL = 10    # 秒
MAX_WORKERS = 64      # 单线程更稳

# =============================
# 下载函数（带自动重试）
# =============================
def safe_download():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n🚀 第 {attempt} 次尝试下载（断点续传中）...")
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                allow_patterns=ALLOW_PATTERNS,
                resume_download=True,  # 自动断点续传
                max_workers=MAX_WORKERS,
                token=TOKEN,
            )
            print("\n✅ 下载完成！文件已保存至：", LOCAL_DIR)
            return True
        except (HfHubHTTPError, ConnectionError, LocalEntryNotFoundError, Exception) as e:
            print(f"⚠️ 第 {attempt} 次下载失败：{type(e).__name__} - {e}")
            print(f"⏳ {RETRY_INTERVAL} 秒后重试...")
            time.sleep(RETRY_INTERVAL)
    print("❌ 达到最大重试次数，仍未完成下载。")
    return False


if __name__ == "__main__":
    print(f"📦 正在下载 Hugging Face 数据集：{REPO_ID}")
    print(f"📁 本地保存路径：{LOCAL_DIR}")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    safe_download()
