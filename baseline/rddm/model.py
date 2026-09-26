"""Region-Disentangled Diffusion Model (RDDM), T=10.

Implements the two predictions in Shome et al. (AAAI 2024): rho predicts the
ROI-guided noisy signal and epsilon predicts ROI noise.  The ROI is derived
from ECG R-peak surrogates during training and is unavailable at sampling.
"""
from __future__ import annotations
import math
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F

def _groups(c, maximum=8):
    for g in range(min(c, maximum), 0, -1):
        if c % g == 0: return g
    return 1

def peak_mask(signal: Tensor, width: int = 15, threshold: float = 0.35,
              dilation: int = 15) -> Tensor:
    if signal.ndim == 3: x = signal[..., 0]
    elif signal.ndim == 2: x = signal
    else: raise ValueError("signal must be [B,L,1] or [B,L]")
    x = x.detach().abs()
    x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
    local = F.max_pool1d(x[:, None], width, stride=1, padding=width//2)[:, 0]
    mask = ((x >= local - 1e-6) & (x > threshold)).float()[:, None]
    mask = F.max_pool1d(mask, dilation, stride=1, padding=dilation//2)[..., :x.shape[-1]]
    return mask[:, 0, :, None]

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        t = t.float().reshape(-1); h = self.dim // 2
        f = torch.exp(-math.log(10000.) * torch.arange(h, device=t.device, dtype=t.dtype) / max(h-1,1))
        return F.pad(torch.cat((torch.sin(t[:,None]*f), torch.cos(t[:,None]*f)), -1), (0, self.dim-2*h))

class Block(nn.Module):
    def __init__(self, c, td):
        super().__init__(); self.n1=nn.GroupNorm(_groups(c),c); self.c1=nn.Conv1d(c,c,7,padding=3); self.t=nn.Linear(td,2*c); self.n2=nn.GroupNorm(_groups(c),c); self.c2=nn.Conv1d(c,c,7,padding=3)
    def forward(self,x,t):
        h=self.n1(x); a,b=self.t(t).chunk(2,-1); h=h*(1+a[:,:,None])+b[:,:,None]
        return x+self.c2(F.silu(self.n2(self.c1(F.silu(h)))))

class RDDMNet(nn.Module):
    def __init__(self, width=64, time_dim=128):
        super().__init__(); self.width=width
        self.te=nn.Sequential(TimeEmbedding(time_dim),nn.Linear(time_dim,2*time_dim),nn.SiLU(),nn.Linear(2*time_dim,time_dim))
        self.inp=nn.Conv1d(1,width,7,padding=3); self.cond=nn.Conv1d(3,width,7,padding=3)
        self.b0=nn.ModuleList([Block(width,time_dim),Block(width,time_dim)])
        self.d1=nn.Conv1d(width,2*width,5,stride=5,padding=2); self.cd1=nn.Conv1d(width,2*width,5,stride=5,padding=2); self.b1=nn.ModuleList([Block(2*width,time_dim),Block(2*width,time_dim)])
        self.d2=nn.Conv1d(2*width,4*width,5,stride=5,padding=2); self.cd2=nn.Conv1d(2*width,4*width,5,stride=5,padding=2); self.b2=nn.ModuleList([Block(4*width,time_dim),Block(4*width,time_dim)])
        self.u1=nn.Conv1d(4*width,2*width,3,padding=1); self.ub1=nn.ModuleList([Block(2*width,time_dim),Block(2*width,time_dim)])
        self.u2=nn.Conv1d(2*width,width,3,padding=1); self.ub2=nn.ModuleList([Block(width,time_dim),Block(width,time_dim)])
        self.out=nn.Sequential(nn.GroupNorm(_groups(width),width),nn.SiLU(),nn.Conv1d(width,1,7,padding=3))
    def _features(self, cond):
        cond=cond.transpose(1,2); diff=F.pad(cond.diff(dim=-1),(1,0)); return torch.cat((cond,diff,peak_mask(cond.transpose(1,2)).transpose(1,2)),1)
    def forward(self,x,t,ppg):
        c=self._features(ppg); te=self.te(t); c0=self.cond(c); h=self.inp(x.transpose(1,2))+c0
        for b in self.b0:h=b(h,te)
        s0=h; c1=self.cd1(c0); h=self.d1(h)+c1
        for b in self.b1:h=b(h,te)
        s1=h; h=self.d2(h)+self.cd2(c1)
        for b in self.b2:h=b(h,te)
        h=F.interpolate(h,size=s1.shape[-1],mode='linear',align_corners=False); h=self.u1(h)+s1
        for b in self.ub1:h=b(h,te)
        h=F.interpolate(h,size=s0.shape[-1],mode='linear',align_corners=False); h=self.u2(h)+s0
        for b in self.ub2:h=b(h,te)
        return self.out(h).transpose(1,2)

class RDDM(nn.Module):
    def __init__(self,width=64,time_dim=128,T=10,roi_width=31,lambda1=100.,lambda2=1.):
        super().__init__(); self.eps=RDDMNet(width,time_dim); self.rho=RDDMNet(width,time_dim); self.T=T; self.roi_width=roi_width; self.lambda1=float(lambda1); self.lambda2=float(lambda2)
        beta=torch.linspace(1e-4, .2, T); self.register_buffer('beta',beta); self.register_buffer('alpha',1-beta); self.register_buffer('abar',torch.cumprod(1-beta,0))
    def loss(self,ppg,ecg):
        b=ecg.shape[0]; ti=torch.randint(1,self.T+1,(b,),device=ecg.device); j=ti-1; ab=self.abar[j].view(-1,1,1); noise=torch.randn_like(ecg); mu=peak_mask(ecg,dilation=self.roi_width); roi_noise=mu*noise
        xt=ab.sqrt()*ecg+(1-ab).sqrt()*noise; xmt=ab.sqrt()*ecg+(1-ab).sqrt()*roi_noise; tau=ti.float()/self.T
        xp=self.rho(xt,tau,ppg); ep=self.eps(xmt,tau,ppg)
        return self.lambda1*F.mse_loss(ep,roi_noise)+self.lambda2*F.mse_loss(xp,xmt)

class RDDMSampler:
    def __init__(self,model,steps=10):
        if steps!=10: raise ValueError('RDDM baseline is defined with T=10')
        self.model=model; self.steps=steps
    @torch.no_grad()
    def __call__(self,ppg,noise=None):
        x=torch.randn_like(ppg) if noise is None else noise.clone()
        for k in range(self.steps,0,-1):
            j=k-1; tau=ppg.new_full((len(ppg),),k/self.steps); xp=self.model.rho(x,tau,ppg); ep=self.model.eps(xp,tau,ppg)
            a=self.model.alpha[j]; ab=self.model.abar[j]; mean=(xp-(1-a).sqrt()/(1-ab).sqrt()*ep)/a.sqrt()
            if k>1: mean=mean+self.model.beta[j].sqrt()*torch.randn_like(x)
            x=mean
        return x
