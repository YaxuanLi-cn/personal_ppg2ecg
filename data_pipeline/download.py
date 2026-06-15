#!/usr/bin/env python3
import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError, HfHubHTTPError

# =============================
# 基本配置
# =============================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # ✅ 使用国内镜像（可选）

REPO_ID = "lucky9-cyou/MIMIC-IV-Waveform"
LOCAL_DIR = "/root/autodl-tmp/personal_ppg2ecg/dataset/mimic-iv-waveform"
ALLOW_PATTERNS = "files/mimic4wdb/0.1.0/*"
TOKEN = ""

# 重试设置
MAX_RETRIES = 9999           # 无限循环（或手动设大一点）
RETRY_INTERVAL = 10         # 每次失败后等待 60 秒再重试
MAX_WORKERS = 10              # 并发线程数（1更稳）
TIMEOUT = 10                 # 单次下载中断检测间隔（秒）

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
                resume_download=True,   # ✅ 自动断点续传
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
    print(f"🎯 允许的文件匹配规则：{ALLOW_PATTERNS}")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    safe_download()


# import os
# import requests
# from getpass import getpass
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin

# def download_physionet_dataset(base_url, save_dir, username=None, password=None):
#     """递归下载 PhysioNet 数据集目录"""
#     os.makedirs(save_dir, exist_ok=True)

#     session = requests.Session()
#     session.auth = (username, password)

#     # 先访问目录，解析文件列表
#     response = session.get(base_url)
#     if response.status_code == 401:
#         print("❌ 登录失败：用户名或密码错误。")
#         return
#     elif response.status_code == 403:
#         print("⚠️ 访问被拒绝（可能未获得 MIMIC 访问授权）。")
#         return
#     elif response.status_code != 200:
#         print(f"❌ 访问失败：HTTP {response.status_code}")
#         return

#     soup = BeautifulSoup(response.text, "html.parser")
#     links = [a["href"] for a in soup.find_all("a") if a["href"] not in ("../",)]

#     for link in links:
#         file_url = urljoin(base_url, link)
#         local_path = os.path.join(save_dir, link)

#         if link.endswith("/"):
#             # 递归下载子目录
#             download_physionet_dataset(file_url, local_path, username, password)
#         else:
#             # 文件下载（带断点续传）
#             print(f"⬇️ 正在下载: {file_url}")
#             headers = {}
#             if os.path.exists(local_path):
#                 headers["Range"] = f"bytes={os.path.getsize(local_path)}-"

#             with session.get(file_url, headers=headers, stream=True) as r:
#                 if r.status_code in (200, 206):
#                     with open(local_path, "ab") as f:
#                         for chunk in r.iter_content(chunk_size=8192):
#                             if chunk:
#                                 f.write(chunk)
#                     print(f"✅ 下载完成: {local_path}")
#                 else:
#                     print(f"⚠️ 跳过: {file_url} (HTTP {r.status_code})")

# if __name__ == "__main__":
#     base_url = "https://physionet.org/files/mimic4wdb/0.1.0/"
#     save_dir = "/root/autodl-tmp/fengyuan/data/mimic4wdb"

#     # username = input("PhysioNet 用户名: ")
#     # password = getpass("PhysioNet 密码: ")
#     username = 'ferry1231'
#     password = '8K4W/SuszY#n!_3'

#     download_physionet_dataset(base_url, save_dir, username, password)
