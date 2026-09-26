"""Train the CardioGAN PPG<->ECG cycle model on the MIMIC tensors."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import MIMICPairDataset
from model import CardioGAN
ROOT=Path(__file__).resolve().parents[2]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def set_requires(module,flag):
    for p in module.parameters(): p.requires_grad_(flag)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default=str(ROOT/'mimic-iv-aligned-ppg_ecgII-processed-filtered')); p.add_argument('--out-dir',default='results/cardiogan'); p.add_argument('--steps',type=int,default=100000); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--base-channels',type=int,default=32); p.add_argument('--depth',type=int,default=6); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--lambda-cycle',type=float,default=30.); p.add_argument('--alpha-time',type=float,default=3.); p.add_argument('--beta-freq',type=float,default=1.); p.add_argument('--num-workers',type=int,default=8); p.add_argument('--seed',type=int,default=42); p.add_argument('--paired',action='store_true'); p.add_argument('--resume',default=''); p.add_argument('--log-every',type=int,default=100); a=p.parse_args()
    seed_all(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); ds=MIMICPairDataset(a.data_dir,'train'); g=torch.Generator().manual_seed(a.seed)
    p_loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,drop_last=True,num_workers=a.num_workers,pin_memory=True,generator=g); e_loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,drop_last=True,num_workers=a.num_workers,pin_memory=True,generator=g); p_it,e_it=iter(p_loader),iter(e_loader)
    model=CardioGAN(a.base_channels,a.depth).to(device); opt_g=torch.optim.Adam(list(model.g_ecg.parameters())+list(model.g_ppg.parameters()),lr=a.lr,betas=(.5,.999)); opt_d=torch.optim.Adam([p for n,p in model.named_parameters() if n.startswith('d_')],lr=a.lr,betas=(.5,.999)); bce=nn.BCEWithLogitsLoss(); start=0
    if a.resume:
        state=torch.load(a.resume,map_location=device,weights_only=False); model.load_state_dict(state['model']); opt_g.load_state_dict(state['opt_g']); opt_d.load_state_dict(state['opt_d']); start=int(state['step'])
    log_path=out/'train_log.jsonl'; t0=time.time()
    for step in range(start+1,a.steps+1):
        if a.paired:
            try: ppg,ecg,_=next(p_it)
            except StopIteration: p_it=iter(p_loader); ppg,ecg,_=next(p_it)
        else:
            try: ppg,_,_=next(p_it)
            except StopIteration: p_it=iter(p_loader); ppg,_,_=next(p_it)
            try: _,ecg,_=next(e_it)
            except StopIteration: e_it=iter(e_loader); _,ecg,_=next(e_it)
        ppg,ecg=ppg.to(device,non_blocking=True),ecg.to(device,non_blocking=True)
        with torch.no_grad(): fake_ecg,fake_ppg=model.g_ecg(ppg),model.g_ppg(ecg)
        opt_d.zero_grad(set_to_none=True); d_loss=0.
        for disc,real,fake,weight in ((model.d_ecg_t,ecg,fake_ecg,a.alpha_time),(model.d_ppg_t,ppg,fake_ppg,a.alpha_time),(model.d_ecg_f,ecg,fake_ecg,a.beta_freq),(model.d_ppg_f,ppg,fake_ppg,a.beta_freq)):
            real_logit,fake_logit=disc(real),disc(fake); d_loss=d_loss+weight*.5*(bce(real_logit,torch.ones_like(real_logit))+bce(fake_logit,torch.zeros_like(fake_logit)))
        d_loss.backward(); opt_d.step()
        for d in (model.d_ecg_t,model.d_ppg_t,model.d_ecg_f,model.d_ppg_f): set_requires(d,False)
        opt_g.zero_grad(set_to_none=True); fake_ecg,fake_ppg=model.g_ecg(ppg),model.g_ppg(ecg); rec_ppg,rec_ecg=model.g_ppg(fake_ecg),model.g_ecg(fake_ppg)
        ge_t,gp_t=model.d_ecg_t(fake_ecg),model.d_ppg_t(fake_ppg); ge_f,gp_f=model.d_ecg_f(fake_ecg),model.d_ppg_f(fake_ppg)
        g_loss=a.alpha_time*(bce(ge_t,torch.ones_like(ge_t))+bce(gp_t,torch.ones_like(gp_t)))+a.beta_freq*(bce(ge_f,torch.ones_like(ge_f))+bce(gp_f,torch.ones_like(gp_f))); cycle=torch.mean(torch.abs(rec_ppg-ppg))+torch.mean(torch.abs(rec_ecg-ecg)); total=g_loss+a.lambda_cycle*cycle; total.backward(); opt_g.step()
        for d in (model.d_ecg_t,model.d_ppg_t,model.d_ecg_f,model.d_ppg_f): set_requires(d,True)
        if step%a.log_every==0 or step==1:
            record={'step':step,'d_loss':float(d_loss),'g_adv':float(g_loss),'cycle':float(cycle),'loss':float(total),'hours':(time.time()-t0)/3600}; log_path.open('a').write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
        if step%5000==0 or step==a.steps: torch.save({'step':step,'model':model.state_dict(),'opt_g':opt_g.state_dict(),'opt_d':opt_d.state_dict(),'args':vars(a)},out/'last.pt')
    torch.save({'step':a.steps,'model':model.state_dict(),'opt_g':opt_g.state_dict(),'opt_d':opt_d.state_dict(),'args':vars(a)},out/'final.pt')
if __name__=='__main__': main()
