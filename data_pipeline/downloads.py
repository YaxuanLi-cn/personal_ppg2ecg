#!/usr/bin/env python3
import os
import re
import json
import time
import urllib.request
import urllib.error

# =============================
# 基本配置
# =============================
# 使用 HF 镜像（国内可访问）。注意：镜像返回的分页 next 链接指向 huggingface.co，
# 国内无法访问，会导致 snapshot_download 在“列举文件”阶段卡死。本脚本自行分页并
# 把 next 链接的域名改写回镜像，从而绕过该问题。
HF_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "lucky9-cyou/MIMIC-IV-Waveform"
REPO_TYPE = "dataset"
REVISION = "main"
LOCAL_DIR = "/root/autodl-tmp/personal_ppg2ecg/dataset/mimic-iv-waveform_"
SUFFIX = ".hea"   # ✅ 只下载 .hea 文件
TOKEN = None      # 公开数据集留空即可；私有数据集填写你的 token 字符串

# 重试设置
LIST_TIMEOUT = 60          # 列举每页的网络超时（秒）
FILE_MAX_RETRIES = 5       # 单个文件下载失败的最大重试次数
RETRY_INTERVAL = 10        # 重试间隔（秒）


def _auth_headers():
    h = {"User-Agent": "hf-mirror-downloader"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


def iter_repo_files():
    """流式列举仓库中所有文件路径。

    自行处理分页，并把镜像返回的指向 huggingface.co 的 next 链接改写回镜像域名，
    避免在国内环境下卡死。
    """
    url = (
        f"{HF_ENDPOINT}/api/{REPO_TYPE}s/{REPO_ID}/tree/{REVISION}"
        f"?recursive=true"
    )
    while url:
        for attempt in range(1, FILE_MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url, headers=_auth_headers())
                resp = urllib.request.urlopen(req, timeout=LIST_TIMEOUT)
                data = json.loads(resp.read())
                link = resp.headers.get("Link", "") or ""
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"⚠️ 列举文件失败（第 {attempt} 次）：{type(e).__name__} - {e}")
                if attempt == FILE_MAX_RETRIES:
                    raise
                time.sleep(RETRY_INTERVAL)

        for entry in data:
            if entry.get("type") == "file":
                yield entry["path"]

        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if m:
            # 关键修复：把 next 链接的域名从 huggingface.co 改回镜像
            url = m.group(1).replace("https://huggingface.co", HF_ENDPOINT)
        else:
            url = None


def download_one(path):
    """下载单个文件（自动断点续传、跳过已存在），带重试。"""
    local_path = os.path.join(LOCAL_DIR, path)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return "skip"
    for attempt in range(1, FILE_MAX_RETRIES + 1):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=path,
                repo_type=REPO_TYPE,
                revision=REVISION,
                local_dir=LOCAL_DIR,
                token=TOKEN,
            )
            return "ok"
        except (HfHubHTTPError, ConnectionError, TimeoutError, OSError) as e:
            print(f"   ⚠️ 下载 {path} 失败（第 {attempt} 次）：{type(e).__name__} - {e}")
            if attempt == FILE_MAX_RETRIES:
                return "fail"
            time.sleep(RETRY_INTERVAL)


if __name__ == "__main__":
    print(f"📦 正在下载 Hugging Face 数据集：{REPO_ID}")
    print(f"📁 本地保存路径：{LOCAL_DIR}")
    print(f"🎯 只下载后缀为 {SUFFIX} 的文件")
    print(f"🌐 使用镜像：{HF_ENDPOINT}（已修复分页 next 链接指向官网导致的卡死问题）\n")
    os.makedirs(LOCAL_DIR, exist_ok=True)

    seen = ok = skipped = failed = 0
    start = time.time()
    for path in iter_repo_files():
        if not path.endswith(SUFFIX):
            continue
        seen += 1
        result = download_one(path)
        if result == "ok":
            ok += 1
        elif result == "skip":
            skipped += 1
        else:
            failed += 1
        if seen % 50 == 0:
            elapsed = round(time.time() - start, 1)
            print(f"📊 已发现 {seen} 个 {SUFFIX} 文件 | 新下载 {ok} | 跳过 {skipped} | 失败 {failed} | 用时 {elapsed}s")

    elapsed = round(time.time() - start, 1)
    print(f"\n✅ 完成！共 {seen} 个 {SUFFIX} 文件 | 新下载 {ok} | 跳过 {skipped} | 失败 {failed} | 用时 {elapsed}s")
    print(f"📁 文件保存在：{LOCAL_DIR}")
