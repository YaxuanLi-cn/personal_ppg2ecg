import os
import numpy as np
import neurokit2 as nk
import mne
from scipy.signal import savgol_filter
import biobss
import argparse
import logging
from typing import Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


# -----------------------------
# Band-pass filtering (added for reproduction)
# The Arrow->npz pipeline (ecg_ppg_pipeline.py) outputs raw resampled segments
# without any band-pass filtering, while the downstream quality checks assume
# cleaned signals. We apply the same filtering used in step1.py's MCMEDProcessor
# (PPG 0.5-8Hz, ECG 0.5-40Hz + 50Hz notch) before smoothing/QC so the quality
# control operates on clean signals.
# -----------------------------
def _bandpass_ppg(x: np.ndarray, sr: float = 125.0,
                  l_freq: float = 0.5, h_freq: float = 8.0) -> np.ndarray:
    return mne.filter.filter_data(
        x.astype(float), sfreq=sr, l_freq=l_freq, h_freq=h_freq,
        method='iir', iir_params=dict(order=3, ftype='butter', output='ba'),
        verbose=False,
    ).astype(np.float32)


def _bandpass_ecg(x: np.ndarray, sr: float = 125.0,
                  l_freq: float = 0.5, h_freq: float = 40.0,
                  notch: float = 50.0) -> np.ndarray:
    y = mne.filter.filter_data(
        x.astype(float), sfreq=sr, l_freq=l_freq, h_freq=h_freq,
        method='iir', iir_params=dict(order=5, ftype='butter', output='ba'),
        verbose=False,
    )
    if notch is not None and notch < sr / 2:
        y = mne.filter.notch_filter(
            y, Fs=sr, freqs=notch, notch_widths=1, method='iir', verbose=False,
        )
    return y.astype(np.float32)


# -----------------------------
# Utility functions
# -----------------------------
def _safe_savgol(x: np.ndarray, window_length: int, polyorder: int):
    """Automatically adjust window length to be odd and not exceed signal length; return original if too short."""
    L = len(x)
    if L < polyorder + 2:
        return x
    w = min(window_length, L if L % 2 == 1 else L - 1)
    if w < polyorder + 2:
        w = polyorder + 3
        if w % 2 == 0:
            w += 1
        if w > L:
            return x
    return savgol_filter(x, window_length=w, polyorder=polyorder)


def _batch_savgol_parallel(arr: np.ndarray, win: int, poly: int, desc: str = "S-G smoothing", n_jobs: int = -1):
    """Parallel version of Savitzky-Golay smoothing"""
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    n_jobs = min(n_jobs, len(arr))  # Don't use more workers than samples
    
    if n_jobs == 1 or len(arr) < 10:  # Fall back to serial for small arrays
        out = np.zeros_like(arr)
        for i in tqdm(range(arr.shape[0]), desc=desc, leave=False):
            out[i] = _safe_savgol(arr[i], win, poly)
        return out
    
    # Parallel processing
    worker_fn = partial(_safe_savgol, window_length=win, polyorder=poly)
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(worker_fn, arr),
            total=len(arr),
            desc=desc,
            leave=False
        ))
    return np.array(results)


# -----------------------------
# Worker functions for parallel processing
# -----------------------------
def _ppg_quality_worker(args):
    """Worker function for PPG quality checking"""
    item, sr, delta, correct_peaks, sim_thre, frac_thre = args
    try:
        baseline_drift = np.abs(np.mean(np.diff(item)))
        baseline_ok = baseline_drift < 0.1

        smooth = _safe_savgol(item, window_length=9, polyorder=2)
        noise = item - smooth
        signal_power = np.mean(item ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0
        snr_ok = snr > 10

        info = biobss.ppgtools.ppg_detectpeaks(sig=item, sampling_rate=sr,
                                               method='peakdet', delta=delta, correct_peaks=correct_peaks)
        locs_peaks = info['Peak_locs']
        if len(locs_peaks) > 1:
            intervals = np.diff(locs_peaks)
            cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
            interval_ok = (cv < 0.5) and (0.5 * sr < np.mean(intervals) < 2.0 * sr)
        else:
            interval_ok = False

        sim, _ = biobss.sqatools.template_matching(item, locs_peaks)
        sim = np.asarray(sim)
        ppg_sqi = (sim >= sim_thre).sum(axis=-1) / (sim.shape[-1] + 1e-8) if sim.ndim > 0 else 0.0
        template_ok = ppg_sqi > frac_thre

        amplitude_ok = 0.1 < (np.max(item) - np.min(item)) < 10.0
        checks = [baseline_ok, snr_ok, interval_ok, template_ok, amplitude_ok]
        sqi_flag = (np.mean(checks) >= 0.8)
    except Exception:
        sqi_flag = False
    return sqi_flag


def _ecg_quality_worker(args):
    """Worker function for ECG quality checking"""
    item, sr, sqi_thre, frac_thre, method = args
    try:
        scores = []
        ecg_cleaned = nk.ecg_clean(item, sampling_rate=sr, method="neurokit")
        rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sr)[1]['ECG_R_Peaks']
        min_expected_peaks = int(len(item) / sr * 0.5)
        scores.append(1.0 if len(rpeaks) >= min_expected_peaks else 0.0)

        if len(rpeaks) > 3:
            rr = np.diff(rpeaks) / sr
            rr_ok = (0.4 < np.mean(rr) < 1.5) and ((np.std(rr) / (np.mean(rr) + 1e-8)) < 0.5)
            scores.append(1.0 if rr_ok else 0.0)
        else:
            scores.append(0.0)

        amp_ok = 0.2 < (np.max(item) - np.min(item)) < 5.0
        scores.append(1.0 if amp_ok else 0.0)

        from scipy.fft import fft
        freqs = np.fft.fftfreq(len(item), d=1/sr)
        spec = np.abs(fft(item))
        valid = (np.abs(freqs) >= 0.5) & (np.abs(freqs) <= 40)
        ratio = (spec[valid].sum() / (spec.sum() + 1e-12)) if spec.sum() > 0 else 0.0
        scores.append(1.0 if ratio > sqi_thre else 0.0)

        return np.mean(scores) >= frac_thre
    except Exception:
        return False


# -----------------------------
# ECG Dataset + Quality Checks (Parallel version)
# -----------------------------
class ECGDataset:
    def __init__(self, file_path: str):
        data = np.load(file_path, allow_pickle=True)
        self.file_name = data["file_name"]
        if np.isscalar(self.file_name) or np.ndim(self.file_name) == 0:
            self.file_name = np.array([self.file_name])
        self.ppg_data = data["PPG"]
        self.ecg_data = data["ECG"]
        if 'ECG_I' in data:
            self.ecg_I_data = data['ECG_I']

        # ensure shape (N, L)
        if self.ppg_data.ndim == 1: self.ppg_data = self.ppg_data[None, :]
        if self.ecg_data.ndim == 1: self.ecg_data = self.ecg_data[None, :]
        if hasattr(self, 'ecg_I_data') and self.ecg_I_data.ndim == 1:
            self.ecg_I_data = self.ecg_I_data[None, :]

    def __len__(self):
        return len(self.ecg_data)

    def ppg_quality_checker(self, data: np.ndarray, sr: float, delta: float = 1e-4,
                            correct_peaks: bool = True, sim_thre: float = 0.8, 
                            frac_thre: float = 0.8, n_jobs: int = -1):
        """Parallel PPG quality checking"""
        assert data.ndim in (1, 2)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        if n_jobs == -1:
            n_jobs = cpu_count()
        
        n_jobs = min(n_jobs, len(data))
        
        if n_jobs == 1 or len(data) < 10:  # Serial for small datasets
            check_result = []
            for item in tqdm(data, desc="PPG Quality", leave=False):
                args = (item, sr, delta, correct_peaks, sim_thre, frac_thre)
                check_result.append(_ppg_quality_worker(args))
            return np.asarray(check_result, dtype=bool)
        
        # Parallel processing
        args_list = [(item, sr, delta, correct_peaks, sim_thre, frac_thre) for item in data]
        with Pool(processes=n_jobs) as pool:
            check_result = list(tqdm(
                pool.imap(_ppg_quality_worker, args_list),
                total=len(args_list),
                desc="PPG Quality",
                leave=False
            ))
        
        return np.asarray(check_result, dtype=bool)

    def ecg_quality_checker(self, data: np.ndarray, sr: float,
                            sqi_thre: float = 0.95, frac_thre: float = 0.5, 
                            method: str = 'custom', n_jobs: int = -1):
        """Parallel ECG quality checking"""
        assert data.ndim in (1, 2)
        assert method in ['custom', 'neurokit']
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        if n_jobs == -1:
            n_jobs = cpu_count()
        
        n_jobs = min(n_jobs, len(data))
        
        if n_jobs == 1 or len(data) < 10:  # Serial for small datasets
            check_result = []
            for item in tqdm(data, desc="ECG Quality", leave=False):
                args = (item, sr, sqi_thre, frac_thre, method)
                check_result.append(_ecg_quality_worker(args))
            return np.asarray(check_result, dtype=bool)
        
        # Parallel processing
        args_list = [(item, sr, sqi_thre, frac_thre, method) for item in data]
        with Pool(processes=n_jobs) as pool:
            check_result = list(tqdm(
                pool.imap(_ecg_quality_worker, args_list),
                total=len(args_list),
                desc="ECG Quality",
                leave=False
            ))
        
        return np.asarray(check_result, dtype=bool)


# -----------------------------
# Main pipeline
# -----------------------------
def process_and_save(input_npz: str,
                     output_npz: Optional[str] = None,
                     orig_sr: float = 125.0,
                     target_sr: float = 125.0,
                     ppg_win: int = 7,
                     ppg_poly: int = 2,
                     ecg_win: int = 11,
                     ecg_poly: int = 2,
                     ecg_method: str = 'custom',
                     n_jobs: int = -1):
    logging.info(f"Loading: {input_npz}")
    ds = ECGDataset(input_npz)
    file_name = ds.file_name
    PPG = ds.ppg_data.astype(np.float32)
    ECG = ds.ecg_data.astype(np.float32)
    ECG_I = getattr(ds, 'ecg_I_data', None)

    # Band-pass filtering (added for reproduction; see _bandpass_* above)
    logging.info("Applying band-pass filtering ...")
    PPG = _bandpass_ppg(PPG, sr=orig_sr)
    ECG = _bandpass_ecg(ECG, sr=orig_sr)
    if ECG_I is not None:
        ECG_I = _bandpass_ecg(ECG_I.astype(np.float32), sr=orig_sr)

    # S-G smoothing (parallel)
    logging.info("Applying Savitzky–Golay smoothing ...")
    PPG_s = _batch_savgol_parallel(PPG, ppg_win, ppg_poly, desc="PPG smoothing", n_jobs=n_jobs)
    ECG_s = _batch_savgol_parallel(ECG, ecg_win, ecg_poly, desc="ECG smoothing", n_jobs=n_jobs)
    ECG_I_s = _batch_savgol_parallel(ECG_I, ecg_win, ecg_poly, desc="ECG_I smoothing", n_jobs=n_jobs) if ECG_I is not None else None

    # Quality control (parallel)
    logging.info("Running quality checks ...")
    ppg_flags = ds.ppg_quality_checker(PPG_s, sr=target_sr, n_jobs=n_jobs)
    ecg_flags = ds.ecg_quality_checker(ECG_s, sr=target_sr, method=ecg_method, n_jobs=n_jobs)
    if ECG_I_s is not None:
        ecg_I_flags = ds.ecg_quality_checker(ECG_I_s, sr=target_sr, method=ecg_method, n_jobs=n_jobs)
        ecg_flags &= ecg_I_flags

    # Apply mask
    keep_mask = ppg_flags & ecg_flags
    kept = int(keep_mask.sum())
    total = len(keep_mask)
    logging.info(f"Kept {kept}/{total} samples ({kept/total:.1%}).")

    # Save
    file_name = file_name[keep_mask]
    PPG_kept = PPG_s[keep_mask]
    ECG_kept = ECG_s[keep_mask]
    ECG_I_kept = ECG_I_s[keep_mask] if ECG_I_s is not None else None

    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
    save_dict = {'file_name': file_name, 'PPG': PPG_kept, 'ECG': ECG_kept}
    if ECG_I_kept is not None:
        save_dict['ECG_I'] = ECG_I_kept
    np.savez_compressed(output_npz, **save_dict)
    logging.info(f"Saved to: {output_npz}")
    return output_npz, kept, total


def process_single_file(args):
    """Worker function for processing a single file"""
    input_path, output_path, n_jobs = args
    try:
        process_and_save(input_npz=input_path, output_npz=output_path, n_jobs=n_jobs)
        return True
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        return False


def parse_args():
    p = argparse.ArgumentParser(description="PPG/ECG quality filtering with parallel processing")
    p.add_argument("--input", type=str, required=True, help="Input directory with .npz files")
    p.add_argument("--output", type=str, required=True, help="Output directory")
    p.add_argument("--n_jobs", type=int, default=-1, 
                   help="Number of parallel jobs (-1 for all CPUs, 1 for serial)")
    p.add_argument("--file_parallel", action="store_true",
                   help="Process multiple files in parallel (may use more memory)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    args = parse_args()
    input_paths = sorted([
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".npz")
    ])
    os.makedirs(args.output, exist_ok=True)

    if args.file_parallel and len(input_paths) > 1:
        # Process files in parallel
        logging.info(f"Processing {len(input_paths)} files in parallel mode")
        file_args = [(inp, os.path.join(args.output, os.path.basename(inp)), 1) 
                     for inp in input_paths]
        
        n_file_workers = min(cpu_count() // 2, len(input_paths))  # Use half CPUs for file-level parallelism
        with Pool(processes=n_file_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, file_args),
                total=len(file_args),
                desc="Processing files"
            ))
        logging.info(f"Successfully processed {sum(results)}/{len(results)} files")
    else:
        # Process files sequentially but parallelize within each file
        logging.info(f"Processing {len(input_paths)} files sequentially (sample-level parallelism)")
        for input_path in tqdm(input_paths, desc="Processing files"):
            output_path = os.path.join(args.output, os.path.basename(input_path))
            process_and_save(input_npz=input_path, output_npz=output_path, n_jobs=args.n_jobs)


'''
使用示例：

# 默认模式：顺序处理文件，每个文件内部并行处理样本（推荐）
python step2_fast.py \
    --input /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed \
    --output /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed-filtered

# 指定使用的CPU核心数
python step2_fast.py \
    --input /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed \
    --output /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed-filtered \
    --n_jobs 8

# 文件级并行模式（适合文件很多但每个文件较小的情况，会占用更多内存）
python step2_fast.py \
    --input /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed \
    --output /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered \
    --file_parallel \
    --n_jobs 8

# 不使用ECG_I的数据
python step2_fast.py \
    --input /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed \
    --output /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered \
    --n_jobs -1
'''