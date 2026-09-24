import os
import numpy as np
import neurokit2 as nk
from scipy.signal import savgol_filter
import biobss
import argparse
import logging
from typing import Optional
from tqdm import tqdm


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


def _batch_savgol(arr: np.ndarray, win: int, poly: int, desc: str = "S-G smoothing"):  # 🔵 tqdm 支持
    out = np.zeros_like(arr)
    for i in tqdm(range(arr.shape[0]), desc=desc, leave=False):
        out[i] = _safe_savgol(arr[i], win, poly)
    return out


# -----------------------------
# ECG Dataset + Quality Checks
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

    # ---------- PPG quality control ----------
    def ppg_quality_checker(self, data: np.ndarray, sr: float, delta: float = 1e-4,
                            correct_peaks: bool = True, sim_thre: float = 0.8, frac_thre: float = 0.8):
        assert data.ndim in (1, 2)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        check_result = []
        for item in tqdm(data, desc="PPG Quality", leave=False):  # 🔵 tqdm for long loop
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
            check_result.append(sqi_flag)
        return np.asarray(check_result, dtype=bool)

    # ---------- ECG quality control ----------
    def ecg_quality_checker(self, data: np.ndarray, sr: float,
                            sqi_thre: float = 0.95, frac_thre: float = 0.8, method: str = 'custom'):
        assert data.ndim in (1, 2)
        assert method in ['custom', 'neurokit']
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        check_result = []
        for item in tqdm(data, desc="ECG Quality", leave=False):  # 🔵 tqdm
            try:
                scores = []
                ecg_cleaned = nk.ecg_clean(item, sampling_rate=sr, method="neurokit")
                rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sr)[1]['ECG_R_Peaks']
                min_expected_peaks = int(len(item) / sr * 0.5)
                scores.append(1.0 if len(rpeaks) >= min_expected_peaks else 0.0)

                if len(rpeaks) > 3:
                    rr = np.diff(rpeaks) / sr
                    rr_ok = (0.4 < np.mean(rr) < 1.5) and ((np.std(rr) / (np.mean(rr) + 1e-8)) < 0.2)
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

                check_result.append(np.mean(scores) >= frac_thre)
            except Exception:
                check_result.append(False)

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
                     ecg_method: str = 'custom'):
    logging.info(f"Loading: {input_npz}")
    ds = ECGDataset(input_npz)
    file_name = ds.file_name
    PPG = ds.ppg_data.astype(np.float32)
    ECG = ds.ecg_data.astype(np.float32)
    ECG_I = getattr(ds, 'ecg_I_data', None)

    # S-G smoothing 🔵
    logging.info("Applying Savitzky–Golay smoothing ...")
    PPG_s = _batch_savgol(PPG, ppg_win, ppg_poly, desc="PPG smoothing")
    ECG_s = _batch_savgol(ECG, ecg_win, ecg_poly, desc="ECG smoothing")
    ECG_I_s = _batch_savgol(ECG_I, ecg_win, ecg_poly, desc="ECG_I smoothing") if ECG_I is not None else None

    # Quality control 🔵
    logging.info("Running quality checks ...")
    ppg_flags = ds.ppg_quality_checker(PPG_s, sr=target_sr)
    ecg_flags = ds.ecg_quality_checker(ECG_s, sr=target_sr, method=ecg_method)
    if ECG_I_s is not None:
        ecg_I_flags = ds.ecg_quality_checker(ECG_I_s, sr=target_sr, method=ecg_method)
        ecg_flags &= ecg_I_flags

    # Apply mask
    keep_mask = ppg_flags & ecg_flags
    kept = int(keep_mask.sum())
    total = len(keep_mask)
    logging.info(f"Kept {kept}/{total} samples ({kept/total:.1%}).")

    # Save
    # print(file_name.shape, file_name[0].shape)
    file_name = file_name[keep_mask]
    PPG_kept = PPG_s[keep_mask]
    ECG_kept = ECG_s[keep_mask]
    ECG_I_kept = ECG_I_s[keep_mask] if ECG_I_s is not None else None

    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
    np.savez_compressed(output_npz, file_name=file_name, PPG=PPG_kept, ECG=ECG_kept, ECG_I=ECG_I_kept)
    logging.info(f"Saved to: {output_npz}")
    return output_npz, kept, total


def parse_args():
    p = argparse.ArgumentParser(description="PPG/ECG quality filtering with progress bars")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_paths = sorted([
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".npz")
    ])
    os.makedirs(args.output, exist_ok=True)

    for input_path in tqdm(input_paths, desc="Processing files"):
        output_path = os.path.join(args.output, os.path.basename(input_path))
        process_and_save(input_npz=input_path, output_npz=output_path)


'''
python step2.py \
    --input /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed \
    --output /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg-ecgI_ecgII-processed-filtered

python step2.py \
    --input /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed \
    --output /root/autodl-tmp/fengyuan/data/mimic-iv-aligned-ppg_ecgII-processed-filtered
'''