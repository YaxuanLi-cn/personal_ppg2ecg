#!/usr/bin/env python3
"""
基于已下载的 .hea 头文件，补下对应的 .dat 信号文件。

思路：
1. 遍历本地 waves 目录里的所有段头文件（如 88399302_0001.hea）。
2. 解析头文件，找出每个段引用的 .dat 文件及其包含的信号名。
3. 只保留同时包含 ECG(II) 和 PPG(Pleth) 的段，下载这些段对应的 .dat。
   （默认跳过 Resp 的 r.dat，可用 --all-dat 下载全部 .dat）
4. 已存在且非空的 .dat 自动跳过，支持断点续传与重试。

用法：
    python download_dat.py                       # 只下含 II+Pleth 段的 e.dat/p.dat
    python download_dat.py --all-dat             # 下载所有引用到的 .dat
    python download_dat.py --waves-dir /path     # 指定 waves 根目录
    python download_dat.py --dry-run             # 只统计，不下载
"""
import os
import re
import time
import argparse

# 使用国内镜像，避免分页 next 链接指向官网导致卡死
HF_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError

REPO_ID = "lucky9-cyou/MIMIC-IV-Waveform"
REPO_TYPE = "dataset"
REVISION = "main"

# 本地 waves 根目录（即从原数据集中抽出来的 waves 文件夹）
DEFAULT_WAVES_DIR = "/root/autodl-tmp/personal_ppg2ecg/waves"

# 仓库内 waves 所在的前缀候选（脚本会自动探测哪个可用）
REPO_PREFIX_CANDIDATES = [
    "files/mimic4wdb/0.1.0/waves",
    "waves",
    "",
]

TOKEN = None              # 公开数据集留空
FILE_MAX_RETRIES = 5      # 单文件最大重试次数
RETRY_INTERVAL = 10       # 重试间隔（秒）

# 需要的信号名（用于筛选段）
REQUIRED_SIGNALS = {"II", "Pleth"}


def parse_segment_header(hea_path):
    """解析一个段头文件，返回 {dat_filename: set(signal_names)}。

    段头文件的信号行格式：
        <dat_filename> <format> <gain> <bits> <baseline> ... <signal_name>
    其中第一个 token 是 .dat 文件名（layout 段为 '~'，跳过），最后一个 token 是信号名。
    """
    dat_to_signals = {}
    try:
        with open(hea_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return dat_to_signals

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if idx == 0:
            # 第一行是 record 行，跳过
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        dat_name = tokens[0]
        sig_name = tokens[-1]
        if dat_name == "~" or not dat_name.endswith(".dat"):
            continue
        dat_to_signals.setdefault(dat_name, set()).add(sig_name)
    return dat_to_signals


def is_segment_header(filename):
    """段头文件形如 <record>_NNNN.hea；顶层 <record>.hea 也可能直接引用 .dat。"""
    return filename.endswith(".hea")


def collect_dat_files(waves_dir, all_dat=False):
    """遍历所有 .hea，收集需要下载的 .dat 的本地路径集合。"""
    needed = set()
    n_hea = 0
    n_seg_match = 0
    for root, _dirs, files in os.walk(waves_dir):
        for fn in files:
            if not is_segment_header(fn):
                continue
            n_hea += 1
            hea_path = os.path.join(root, fn)
            dat_to_signals = parse_segment_header(hea_path)
            if not dat_to_signals:
                continue

            # 该段包含的所有信号
            all_signals = set().union(*dat_to_signals.values())

            if all_dat:
                for dat_name in dat_to_signals:
                    needed.add(os.path.join(root, dat_name))
                continue

            # 只保留同时含 II 和 Pleth 的段
            if REQUIRED_SIGNALS.issubset(all_signals):
                n_seg_match += 1
                for dat_name, sigs in dat_to_signals.items():
                    # 只下含所需信号的 .dat（e.dat 含 II，p.dat 含 Pleth）
                    if sigs & REQUIRED_SIGNALS:
                        needed.add(os.path.join(root, dat_name))
    return needed, n_hea, n_seg_match


def detect_repo_prefix(sample_rel):
    """用一个样本相对路径（如 p103/.../xxx_0001e.dat）探测可用的仓库前缀。"""
    for prefix in REPO_PREFIX_CANDIDATES:
        repo_path = f"{prefix}/{sample_rel}" if prefix else sample_rel
        for attempt in range(1, FILE_MAX_RETRIES + 1):
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=repo_path,
                    repo_type=REPO_TYPE,
                    revision=REVISION,
                    local_dir=_STAGING_DIR,
                    token=TOKEN,
                )
                print(f"✅ 探测到可用仓库前缀：'{prefix}'")
                return prefix
            except EntryNotFoundError:
                break  # 这个前缀不对，换下一个
            except (HfHubHTTPError, ConnectionError, TimeoutError, OSError) as e:
                print(f"   ⚠️ 探测前缀 '{prefix}' 第 {attempt} 次失败：{type(e).__name__} - {e}")
                if attempt == FILE_MAX_RETRIES:
                    break
                time.sleep(RETRY_INTERVAL)
    raise RuntimeError("无法探测到可用的仓库前缀，请检查 REPO_PREFIX_CANDIDATES 或网络。")


def download_one(repo_path, local_path):
    """下载单个 .dat 到 staging，再移动到 local_path（与其 .hea 同目录）。"""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return "skip"
    for attempt in range(1, FILE_MAX_RETRIES + 1):
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                repo_type=REPO_TYPE,
                revision=REVISION,
                local_dir=_STAGING_DIR,
                token=TOKEN,
            )
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                os.replace(downloaded, local_path)
            except OSError:
                import shutil
                shutil.move(downloaded, local_path)
            return "ok"
        except EntryNotFoundError:
            print(f"   ❓ 仓库中不存在：{repo_path}")
            return "missing"
        except (HfHubHTTPError, ConnectionError, TimeoutError, OSError) as e:
            print(f"   ⚠️ 下载 {repo_path} 失败（第 {attempt} 次）：{type(e).__name__} - {e}")
            if attempt == FILE_MAX_RETRIES:
                return "fail"
            time.sleep(RETRY_INTERVAL)


_STAGING_DIR = None  # 运行时设置：HF 下载暂存目录


def main():
    global _STAGING_DIR
    parser = argparse.ArgumentParser(description="补下 MIMIC-IV waveform 的 .dat 信号文件")
    parser.add_argument("--waves-dir", default=DEFAULT_WAVES_DIR, help="本地 waves 根目录")
    parser.add_argument("--all-dat", action="store_true", help="下载所有引用到的 .dat（含 Resp 等）")
    parser.add_argument("--dry-run", action="store_true", help="只统计需要下载的文件，不实际下载")
    args = parser.parse_args()

    waves_dir = os.path.abspath(args.waves_dir)
    if not os.path.isdir(waves_dir):
        raise SystemExit(f"waves 目录不存在：{waves_dir}")

    # HF 下载暂存目录（下完后移动到各自 .hea 同级目录）
    _STAGING_DIR = os.path.join(os.path.dirname(waves_dir), ".hf_dat_staging")
    os.makedirs(_STAGING_DIR, exist_ok=True)

    print(f"📂 扫描 .hea：{waves_dir}")
    needed_local, n_hea, n_seg_match = collect_dat_files(waves_dir, all_dat=args.all_dat)
    print(f"   共扫描 {n_hea} 个 .hea；匹配段 {n_seg_match}；需要 {len(needed_local)} 个 .dat")

    if not needed_local:
        print("没有需要下载的 .dat。")
        return

    # 计算每个本地 .dat 相对 waves_dir 的相对路径（即仓库 waves 前缀之后的部分）
    rel_paths = {lp: os.path.relpath(lp, waves_dir) for lp in needed_local}  # p103/.../xxx_0001e.dat

    if args.dry_run:
        for lp in sorted(needed_local)[:20]:
            print("  ", rel_paths[lp])
        print(f"... (dry-run) 共 {len(needed_local)} 个文件")
        return

    # 探测仓库前缀（前缀候选已包含 waves，故直接拼相对路径）
    repo_prefix = detect_repo_prefix(next(iter(rel_paths.values())))

    ok = skipped = failed = missing = 0
    start = time.time()
    for i, (lp, rel) in enumerate(sorted(rel_paths.items()), 1):
        repo_path = f"{repo_prefix}/{rel}" if repo_prefix else rel
        result = download_one(repo_path, lp)
        if result == "ok":
            ok += 1
        elif result == "skip":
            skipped += 1
        elif result == "missing":
            missing += 1
        else:
            failed += 1
        if i % 50 == 0:
            elapsed = round(time.time() - start, 1)
            print(f"📊 进度 {i}/{len(rel_paths)} | 新下载 {ok} | 跳过 {skipped} | 缺失 {missing} | 失败 {failed} | 用时 {elapsed}s")

    elapsed = round(time.time() - start, 1)
    print(f"\n✅ 完成！需要 {len(rel_paths)} 个 .dat | 新下载 {ok} | 跳过 {skipped} | 缺失 {missing} | 失败 {failed} | 用时 {elapsed}s")


if __name__ == "__main__":
    main()
