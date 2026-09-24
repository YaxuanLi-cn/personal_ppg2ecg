#!/usr/bin/env python3
import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import numpy as np
from scipy.signal import resample
import wfdb


ROOT = Path(__file__).resolve().parents[1]
FS = 125
WINDOW = 1250


def parse_args():
    parser = argparse.ArgumentParser(
        description="WFDB -> paired 125 Hz / 10 s NPZ -> quality filtering -> train/test PT."
    )
    parser.add_argument("--input", type=Path, default=ROOT / "MIMIC-IV-Waveform/files/mimic4wdb/0.1.0/waves")
    parser.add_argument("--output", type=Path, default=ROOT / "mimic-iv-aligned-ppg_ecgII-processed-filtered",
                        help="New output directory; existing nonempty directories are refused")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-seconds", type=int, default=300)
    parser.add_argument("--max-subjects", type=int, default=0, help="0 = all; use a separate output for smoke tests")
    parser.add_argument("--max-windows-per-subject", type=int, default=0, help="Limit pre-QC windows; 0 = all")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.chunk_seconds < 10:
        parser.error("--n-jobs must be >= 1 and --chunk-seconds must be >= 10")
    if not 0 < args.test_ratio < 1:
        parser.error("--test-ratio must be between 0 and 1")
    if args.max_subjects < 0 or args.max_windows_per_subject < 0:
        parser.error("Sample limits must be >= 0")
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    return args


def paired_windows(record):
    channels = [record.sig_name.index(name) for name in ("II", "Pleth")]
    signals = [np.asarray(record.e_p_signal[i]) for i in channels]
    spf = [record.samps_per_frame[i] for i in channels]
    frames = min(len(x) // count for x, count in zip(signals, spf))
    valid = np.ones(frames, dtype=bool)
    for x, count in zip(signals, spf):
        valid &= np.isfinite(x[:frames * count]).reshape(frames, count).all(axis=1)
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        length = int((stop - start) * FS / record.fs)
        windows = length // WINDOW
        if not windows:
            continue
        arrays = [
            resample(x[start * count:stop * count], length).astype(np.float32)[:windows * WINDOW].reshape(-1, WINDOW)
            for x, count in zip(signals, spf)
        ]
        ecg, ppg = arrays
        keep = (np.ptp(ecg, axis=1) > 1e-8) & (np.ptp(ppg, axis=1) > 1e-8)
        keep &= np.isfinite(ecg).all(axis=1) & np.isfinite(ppg).all(axis=1)
        if keep.any():
            yield ppg[keep], ecg[keep]


def convert_subject(subject, raw_dir, args):
    ppg_parts, ecg_parts = [], []
    count = errors = 0
    for header in sorted(subject.glob("*/*.hea")):
        try:
            metadata = wfdb.rdheader(str(header.with_suffix("")))
        except Exception as exc:
            logging.warning("Unreadable header %s: %s", header, exc)
            errors += 1
            continue
        if not isinstance(metadata, wfdb.Record) or not metadata.sig_len:
            continue
        if not {"II", "Pleth"}.issubset(metadata.sig_name):
            continue
        if metadata.sig_len / metadata.fs < 10:
            continue
        chunk = max(1, int(np.ceil(metadata.fs * args.chunk_seconds)))
        logging.info("Reading %s (%.1f seconds)", header.name, metadata.sig_len / metadata.fs)
        for start in range(0, metadata.sig_len, chunk):
            stop = min(start + chunk, metadata.sig_len)
            if (stop - start) / metadata.fs < 10:
                continue
            try:
                record = wfdb.rdrecord(
                    str(header.with_suffix("")), sampfrom=start, sampto=stop,
                    channel_names=["II", "Pleth"], smooth_frames=False,
                )
            except Exception as exc:
                logging.warning("Skipping unreadable block %s [%d:%d]: %s", header.name, start, stop, exc)
                errors += 1
                continue
            for ppg, ecg in paired_windows(record):
                if args.max_windows_per_subject:
                    remaining = args.max_windows_per_subject - count
                    ppg, ecg = ppg[:remaining], ecg[:remaining]
                ppg_parts.append(ppg)
                ecg_parts.append(ecg)
                count += len(ppg)
                if args.max_windows_per_subject and count >= args.max_windows_per_subject:
                    break
            if args.max_windows_per_subject and count >= args.max_windows_per_subject:
                break
        if args.max_windows_per_subject and count >= args.max_windows_per_subject:
            break
    if count:
        np.savez_compressed(
            raw_dir / f"{subject.name}.npz",
            PPG=np.concatenate(ppg_parts), ECG=np.concatenate(ecg_parts),
            file_name=np.full(count, subject.name),
        )
    logging.info("%s: %d paired windows, %d read errors", subject.name, count, errors)
    return count, errors


def run_stage(script, arguments, logfile):
    command = [sys.executable, "-u", str(script), *map(str, arguments)]
    logging.info("Running: %s", " ".join(command))
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, env=env) as process:
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logfile.write(line)
                logfile.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def validate_output(output):
    import torch

    with (output / "z_score_mean_std.json").open() as stream:
        stats = json.load(stream)["train"]
    for name in ("PPG", "ECG"):
        if not np.isfinite(stats[name]["mean"]) or not np.isfinite(stats[name]["std"]) or stats[name]["std"] <= 0:
            raise ValueError(f"Invalid normalization statistics for {name}: {stats[name]}")
    sizes = {}
    for split in ("train", "test"):
        data = torch.load(output / f"{split}.pt", map_location="cpu", weights_only=True)
        size = len(data["PPG"])
        if data["PPG"].shape != (size, WINDOW) or data["ECG"].shape != (size, WINDOW):
            raise ValueError(f"Unexpected signal shapes in {split}.pt")
        if data["file_name"].shape != (size,) or data["file_name"].dtype != torch.int64:
            raise ValueError(f"Invalid subject IDs in {split}.pt")
        for name in ("PPG", "ECG"):
            if data[name].dtype != torch.float32:
                raise ValueError(f"Invalid {name} dtype in {split}.pt")
            for batch in data[name].split(4096):
                if not torch.isfinite(batch).all():
                    raise ValueError(f"Nonfinite {name} values in {split}.pt")
        if split == "train" and not size:
            raise ValueError("No training samples survived quality control")
        sizes[split] = size
        logging.info("Validated %s.pt: %d samples, shape (%d, %d)", split, size, size, WINDOW)
        del data
    if not sizes["test"]:
        logging.warning("Test set is empty; use more data before evaluating the model")
    return sizes


def main():
    args = parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input}")
    subjects = sorted(p for p in args.input.glob("p*/p*") if p.is_dir() and p.name[1:].isdigit())
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    if not subjects:
        raise SystemExit("No subject directories found; --input must point to the waves directory")
    if args.output == args.input or args.output in args.input.parents or args.input in args.output.parents:
        raise SystemExit("Output must be separate from the input data tree")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise SystemExit(f"Refusing to overwrite nonempty output: {args.output}; choose a new --output")
    for module in ("torch", "mne", "neurokit2", "biobss"):
        importlib.import_module(module)
    scripts = ROOT / "v0/data_process_to_npz"
    for name in ("step2_fast.py", "build_pt_dataset.py"):
        if not (scripts / name).is_file():
            raise SystemExit(f"Required repository script is missing: {scripts / name}")
    raw_dir = args.output / "raw_npz"
    raw_dir.mkdir(parents=True, exist_ok=False)
    with (args.output / "prepare.log").open("a", buffering=1) as logfile:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.StreamHandler(sys.stdout), logging.StreamHandler(logfile)], force=True)
        try:
            logging.info("Input: %s; output: %s; subjects: %d", args.input, args.output, len(subjects))
            logging.info("Split policy: random windows WITHIN each subject, not held-out subjects")
            logging.info("Stage 1/3: WFDB -> finite, nonflat, 125 Hz / 10 s paired windows")
            total = errors = 0
            for index, subject in enumerate(subjects, 1):
                logging.info("Subject %d/%d: %s", index, len(subjects), subject.name)
                count, failures = convert_subject(subject, raw_dir, args)
                total += count
                errors += failures
            if not total:
                raise ValueError("No paired windows found; inspect II/Pleth channels and read errors")
            logging.info("Stage 2/3: band-pass filtering, smoothing and quality control")
            run_stage(scripts / "step2_fast.py", ["--input", raw_dir, "--output", args.output,
                                                  "--n_jobs", args.n_jobs], logfile)
            usable = 0
            for path in args.output.glob("*.npz"):
                with np.load(path, allow_pickle=False) as data:
                    size = len(data["PPG"])
                    usable += size if size >= 2 else 0
            if not usable:
                raise ValueError("No subject has >= 2 windows after QC; inspect prepare.log or expand the input")
            logging.info("Stage 3/3: building train.pt, test.pt and training-only normalization statistics")
            run_stage(scripts / "build_pt_dataset.py", ["--input", args.output,
                          "--test_ratio", args.test_ratio, "--seed", args.seed], logfile)
            sizes = validate_output(args.output)
            summary = {
                "completed_at": datetime.now().isoformat(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "subjects_scanned": len(subjects), "raw_windows": total, "read_errors": errors,
                "split_policy": "random windows within each subject", "samples": sizes,
            }
            with (args.output / "summary.json").open("w") as stream:
                json.dump(summary, stream, indent=2)
            if errors:
                logging.warning("Completed with %d read errors; unreadable headers/blocks were skipped", errors)
            logging.info("DONE. Set data_dir in v0/model/cardioalign_encoder/train.py to: %s", args.output)
        except Exception:
            logging.exception("Preprocessing failed; partial output retained for inspection. Use a new --output for retry.")
            raise


if __name__ == "__main__":
    main()
