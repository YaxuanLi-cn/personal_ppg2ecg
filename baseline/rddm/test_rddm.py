import unittest, torch
from .model import RDDM,RDDMSampler,peak_mask
class TestRDDM(unittest.TestCase):
 def test_shapes_and_loss(self):
  m=RDDM(width=16,time_dim=32); p=torch.randn(2,1250,1); e=torch.randn_like(p); self.assertTrue(torch.isfinite(m.loss(p,e))); y=RDDMSampler(m)(p,torch.randn_like(p)); self.assertEqual(y.shape,p.shape)
 def test_mask(self): self.assertEqual(peak_mask(torch.randn(2,1250,1)).shape,(2,1250,1))
if __name__=='__main__': unittest.main()
