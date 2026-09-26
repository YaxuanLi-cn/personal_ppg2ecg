"""Generate evaluation arrays from a CardioGAN checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from data import MIMICPairDataset
from model import CardioGAN
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default=str(ROOT/'mimic-iv-aligned-ppg_ecgII-processed-filtered')); p.add_argument('--ckpt',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--indices',default=''); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--num-workers',type=int,default=8); p.add_argument('--base-channels',type=int,default=32); p.add_argument('--depth',type=int,default=6); a=p.parse_args(); out=Path(a.out_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'output must be empty: {out}')
    out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ds=MIMICPairDataset(a.data_dir,'test')
    if a.indices: idx=np.load(a.indices); np.save(out/'subset_indices.npy',idx); view=Subset(ds,idx.tolist())
    else: view=ds
    loader=DataLoader(view,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True); state=torch.load(a.ckpt,map_location=device,weights_only=False); cfg=state.get('args',{}); model=CardioGAN(cfg.get('base_channels',a.base_channels),cfg.get('depth',a.depth)).to(device); model.load_state_dict(state['model']); model.eval(); fake=[];gt=[];ppg=[]
    with torch.no_grad():
        for x,y,_ in loader:
            x=x.to(device,non_blocking=True); fake.append(model.g_ecg(x).squeeze(1).cpu().numpy()); gt.append(y.numpy().squeeze(1)); ppg.append(x.cpu().numpy().squeeze(1))
    nfake,ngt,nppg=np.concatenate(fake),np.concatenate(gt),np.concatenate(ppg); np.save(out/'overall_fake_data.npy',nfake.astype(np.float64)[...,None]); np.save(out/'overall_gt_data.npy',ngt.astype(np.float64)[...,None]); np.save(out/'overall_gt_ppg_data.npy',nppg.astype(np.float64)[...,None]); print(f'saved {len(nfake)} samples to {out}')
if __name__=='__main__': main()
