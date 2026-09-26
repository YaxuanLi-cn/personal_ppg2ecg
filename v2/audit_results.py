import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from personal_ppg2ecg.v2.metrics import finish_metrics, paired_sums

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def source_manifest():
    result = {}
    for version in ("v0", "v1"):
        for path in sorted((REPO / version).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".sh") and "results" not in path.parts:
                result[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "results/source_manifest.json"
    if args.verify_sources:
        saved = json.loads(manifest.read_text())
        current = source_manifest()
        if saved != current:
            raise RuntimeError("v0/v1 sources changed since audit")
        print(f"Verified {len(saved)} v0/v1 source files unchanged", flush=True)
        return
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(source_manifest(), indent=2))
    dirs = {
        "v0": REPO / "v0/results/rectified_flow_personal/mimic-iv-waveform/samples",
        "v1": REPO / "v1/results/rectified_flow_private/mimic-iv-waveform/samples",
    }
    arrays = {key: {name: np.load(path / f"overall_{name}_data.npy", mmap_mode="r")
                    for name in ("gt", "gt_ppg", "fake")} for key, path in dirs.items()}
    if any(a.shape != arrays["v0"]["gt"].shape for item in arrays.values() for a in item.values()):
        raise ValueError("Baseline sample shapes differ")
    totals = {key: np.zeros(3, dtype=np.float64) for key in arrays}
    for start in range(0, len(arrays["v0"]["gt"]), 1024):
        part = slice(start, start + 1024)
        for name in ("gt", "gt_ppg"):
            if not np.array_equal(arrays["v0"][name][part], arrays["v1"][name][part]):
                raise ValueError(f"Different {name} or ordering at {start}")
        for key, item in arrays.items():
            totals[key] += paired_sums(item["gt"][part], item["fake"][part])
    report = {"samples": len(arrays["v0"]["gt"]), "matching_gt_and_ppg_order": True,
              "normalization": "independent per-window z-score, std(ddof=0)+1e-8",
              "results": {key: finish_metrics(*sums) for key, sums in totals.items()}}
    samples = np.load(dirs["v1"] / "overall_fake_samples.npy", mmap_mode="r")
    report["v1_draw_count"] = samples.shape[1]
    report["v1_official_is_first_draw"] = all(
        np.array_equal(samples[start:start + 1024, 0], arrays["v1"]["fake"][start:start + 1024])
        for start in range(0, len(samples), 1024))
    (ROOT / "results/baseline_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
