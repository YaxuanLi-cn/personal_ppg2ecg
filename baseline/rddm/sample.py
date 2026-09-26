from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
try:
 from .data import WaveformDataset
 from .model import RDDM,RDDMSampler
except ImportError:
 from data import WaveformDataset
 from model import RDDM,RDDMSampler
def fixed_noise(start,n,l,seed):
 return torch.from_numpy(np.stack([np.random.default_rng(np.random.SeedSequence([seed,i])).standard_normal((l,1),dtype=np.float32) for i in range(start,start+n)]))
def main():
 p=argparse.ArgumentParser(); root=Path(__file__).resolve().parents[2]; p.add_argument('--data-dir',type=Path,default=root/'mimic-iv-aligned-ppg_ecgII-processed-filtered'); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--workers',type=int,default=4); p.add_argument('--seed',type=int,default=42); p.add_argument('--max-samples',type=int); a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()): raise FileExistsError(f'{out} is non-empty')
 s=torch.load(a.checkpoint,map_location='cpu',weights_only=False); c=s['config']; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=RDDM(int(c['width']),int(c['time_dim']),10,int(c['roi_width']),float(c['lambda1']),float(c['lambda2'])).to(dev); m.load_state_dict(s.get('ema',s['model']),strict=False); m.eval(); ds=WaveformDataset(a.data_dir,'test',a.max_samples); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=dev.type=='cuda',persistent_workers=a.workers>0); n=len(ds); fake=np.lib.format.open_memmap(out/'overall_fake_data.npy',mode='w+',dtype=np.float32,shape=(n,1250,1)); gt=np.lib.format.open_memmap(out/'overall_gt_data.npy',mode='w+',dtype=np.float32,shape=(n,1250,1)); ppgout=np.lib.format.open_memmap(out/'overall_gt_ppg_data.npy',mode='w+',dtype=np.float32,shape=(n,1250,1)); sam=RDDMSampler(m,10); off=0
 with torch.no_grad():
  for ppg,ecg,_ in dl:
   k=len(ppg); pred=sam(ppg.to(dev),fixed_noise(off,k,1250,a.seed).to(dev)).cpu().numpy().astype(np.float32); fake[off:off+k]=pred; gt[off:off+k]=ecg.numpy(); ppgout[off:off+k]=ppg.numpy(); off+=k
   if off%(a.batch_size*10)==0 or off==n: print(f'Generated {off}/{n}',flush=True)
 fake.flush();gt.flush();ppgout.flush(); np.save(out/'target_indices.npy',np.arange(n,dtype=np.int64)); (out/'generation.json').write_text(json.dumps({'checkpoint':str(a.checkpoint.resolve()),'sampling_steps':10,'seed':a.seed,'shape':[n,1250,1]},indent=2)); print(json.dumps({'complete':True,'output':str(out),'shape':[n,1250,1]}),flush=True)
if __name__=='__main__': main()
