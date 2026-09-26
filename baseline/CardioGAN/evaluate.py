"""Run the repository's five legacy metrics on CardioGAN samples."""
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'v4'))
from evaluation.run_eval import evaluate
def main():
    p=argparse.ArgumentParser(); p.add_argument('--samples-dir',required=True); p.add_argument('--sampling-rate',type=int,default=125); p.add_argument('--workers',type=int,default=8); p.add_argument('--output-json',default=''); a=p.parse_args()
    sample_dir=Path(a.samples_dir).resolve(); internal=ROOT/'v4/results/cardiogan_eval_runtime.json'; evaluate(sample_dir,sampling_rate=a.sampling_rate,workers=a.workers,output_json=internal)
    target=Path(a.output_json).resolve() if a.output_json else sample_dir/'five_metrics.json'; target.write_text(internal.read_text()); print(f'wrote {target}')

if __name__ == '__main__':
    main()
