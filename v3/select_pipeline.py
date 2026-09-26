import argparse
import json
from pathlib import Path
import subprocess
import sys

from personal_ppg2ecg.v3.runtime import ROOT, output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='results/spectral_v3')
    parser.add_argument('--cache', default='results/trained_cache')
    parser.add_argument('--output', default='results/selection')
    parser.add_argument('--baseline', default='results/baseline_validation/five_metrics.json')
    args = parser.parse_args()
    run, cache, output = (output_path(path) for path in (args.run, args.cache, args.output))
    if not (run / 'training_complete.json').is_file():
        raise RuntimeError('Training must finish before model selection')
    checkpoint = run / 'best.pt'
    subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'collect.py'), '--checkpoint', str(checkpoint),
                    '--output', str(cache), '--calibration-size', '65536'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'calibrate.py'), '--cache', str(cache),
                    '--output', str(output), '--baseline-json', str(Path(args.baseline).resolve()),
                    '--workers', '8'], cwd=ROOT, check=True)
    selected = json.loads((output / 'selection.json').read_text())
    print(json.dumps({'selection_complete': True, 'validation_regressions_vs_v2': selected['selection_key'][0],
                      'test_evaluated': False, 'checkpoint': str(output / 'inference.pt')}), flush=True)


if __name__ == '__main__':
    main()
