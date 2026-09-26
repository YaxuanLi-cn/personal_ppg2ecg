"""Run the shared five-metric protocol and record a direct v4 comparison."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "v4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", default="results/clepgan/test")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    samples = Path(args.samples_dir).resolve()
    reference = V4 / "results/final_test"
    count = len(np.load(samples / "overall_gt_data.npy", mmap_mode="r"))
    v4_count = len(np.load(reference / "overall_gt_data.npy", mmap_mode="r"))
    if count != v4_count:
        raise ValueError(f"sample count differs from v4: CLEP-GAN={count}, v4={v4_count}")
    for name in ("overall_gt_data.npy", "overall_gt_ppg_data.npy"):
        actual = np.load(samples / name, mmap_mode="r")
        expected = np.load(reference / name, mmap_mode="r")
        if not np.array_equal(actual, expected):
            raise ValueError(f"test ordering or ground truth differs from v4: {name}")

    from evaluation.run_eval import evaluate
    metrics_file = V4 / "results/clepgan/five_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(samples, sampling_rate=125, workers=args.workers, seed=42,
                       output_json=metrics_file)
    v4_report = json.loads((reference / "five_metrics.json").read_text())
    v4_metrics = v4_report["metrics"]
    comparison = {
        "protocol": "shared v4/evaluation/run_eval.py on identical full test rows (125 Hz)",
        "test_count": count,
        "clepgan_metrics": metrics,
        "v4_metrics": v4_metrics,
        "delta_clepgan_minus_v4": {
            key: (None if metrics[key] is None or v4_metrics[key] is None
                  else metrics[key] - v4_metrics[key]) for key in metrics
        },
        "relative_change_percent": {
            key: (None if metrics[key] is None or not v4_metrics[key]
                  else 100 * (metrics[key] - v4_metrics[key]) / abs(v4_metrics[key]))
            for key in metrics
        },
        "metric_direction": "lower_is_better",
        "same_test_targets_and_ground_truth": True,
        "limitations": [
            "CLEP-GAN validation rows were held out from v4's train.pt partition; they are not subject-disjoint.",
            "The v4 test split is within-subject, so this comparison does not reproduce the paper's unseen-subject protocol.",
            "Heart-rate metrics retain the shared legacy 128 Hz fake-only cleaning caveat.",
            "FD uses the shared randomized PCA implementation and has run-to-run variation.",
        ],
    }
    report_path = samples / "comparison_with_v4.json"
    report_path.write_text(json.dumps(comparison, indent=2))
    shutil.copyfile(metrics_file, samples / "five_metrics.json")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
