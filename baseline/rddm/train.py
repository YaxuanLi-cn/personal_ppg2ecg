from __future__ import annotations
import argparse,json,math,random,time
from pathlib import Path
import numpy as np, torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
try:
 from .data import WaveformDataset
 from .model import RDDM
except ImportError:
 from data import WaveformDataset
 from model import RDDM

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
 torch.backends.cudnn.benchmark=True
class EMA:
 def __init__(self,m,d): self.d=d; self.s={n:p.detach().clone() for n,p in m.named_parameters()}
 @torch.no_grad()
 def update(self,m):
  for n,p in m.named_parameters(): self.s[n].mul_(self.d).add_(p.detach(),alpha=1-self.d)
def main():
 p=argparse.ArgumentParser(); root=Path(__file__).resolve().parents[2]
 p.add_argument('--data-dir',type=Path,default=root/'mimic-iv-aligned-ppg_ecgII-processed-filtered'); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=100000); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--workers',type=int,default=4); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--ema-decay',type=float,default=.999); p.add_argument('--save-every',type=int,default=10000); p.add_argument('--log-every',type=int,default=100); p.add_argument('--width',type=int,default=64); p.add_argument('--time-dim',type=int,default=128); p.add_argument('--roi-width',type=int,default=32); p.add_argument('--lambda1',type=float,default=100.); p.add_argument('--lambda2',type=float,default=1.); p.add_argument('--max-samples',type=int); p.add_argument('--seed',type=int,default=42)
 a=p.parse_args(); out=a.output.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()): raise FileExistsError(f'{out} is non-empty')
 seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ds=WaveformDataset(a.data_dir,'train',a.max_samples); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev.type=='cuda',drop_last=len(ds)>=a.batch_size,persistent_workers=a.workers>0)
 m=RDDM(a.width,a.time_dim,10,a.roi_width,a.lambda1,a.lambda2).to(dev); opt=AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=a.lr*.05); ema=EMA(m,a.ema_decay); it=iter(dl); start=time.monotonic(); last=math.nan
 print(json.dumps({'device':str(dev),'train_windows':len(ds),'parameters':sum(x.numel() for x in m.parameters()),'total_steps':a.steps}),flush=True)
 for step in range(1,a.steps+1):
  try: ppg,ecg,_=next(it)
  except StopIteration: it=iter(dl); ppg,ecg,_=next(it)
  ppg,ecg=ppg.to(dev,non_blocking=True),ecg.to(dev,non_blocking=True); opt.zero_grad(set_to_none=True); loss=m.loss(ppg,ecg)
  if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
  loss.backward(); clip_grad_norm_(m.parameters(),1.); opt.step(); sch.step(); ema.update(m); last=float(loss.detach())
  if step%a.log_every==0 or step==1: print(json.dumps({'step':step,'loss':last,'lr':opt.param_groups[0]['lr'],'seconds':time.monotonic()-start}),flush=True)
  if step%a.save_every==0 or step==a.steps:
   state={'model':m.state_dict(),'ema':ema.s,'optimizer':opt.state_dict(),'step':step,'loss':last,'config':vars(a),'protocol':{'name':'region_disentangled_diffusion_model','paper':'Shome et al., AAAI 2024','sampling_steps':10,'beta_schedule':'linear(1e-4,0.2)','roi':'ECG local peak mask','roi_width':a.roi_width,'lambda1':a.lambda1,'lambda2':a.lambda2,'data_rate_hz':125,'window_length':1250}}
   torch.save(state,out/f'checkpoint-{step}.pt'); torch.save(state,out/'last.pt')
 (out/'config.json').write_text(json.dumps(vars(a),default=str,indent=2)); print(json.dumps({'complete':True,'step':a.steps,'loss':last,'checkpoint':str(out/'last.pt')}),flush=True)
if __name__=='__main__': main()
