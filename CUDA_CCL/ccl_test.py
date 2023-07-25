import torch
from torch.utils import cpp_extension
import time
# import cuda_ccl
cuda_module = cpp_extension.load(name="cuda_ccl", sources=["CCL_ext.cpp", "CCL.cu"], verbose=True)
a=torch.zeros([20, 1080, 1920], dtype=torch.bool, device='cuda:0')
b=torch.zeros([20, 1080, 1920], dtype=torch.int32, device='cuda:0')
a[:, 1:5, 5:8]=True
a[:, 10:11, 15:18]=True
torch.cuda.synchronize()
t=time.time()
# cuda_module.cuda_ccl.torch_ccl(b,a,a.shape[1],a.shape[2])
cuda_module.torch_ccl(b,a,a.shape[1],a.shape[2])
torch.cuda.synchronize()
t1=time.time()
tt2=0
for i in range(b.shape[0]):
    c=b[i]
    d=a[i]
    # tlabel, tcount = torch.unique(c[d], return_counts=True, sorted=False)
    # maxlabelcount = torch.argmax(tcount)
    # maxlabel = tlabel[maxlabelcount].item()
    # print(tlabel, tcount, torch.argmax(tcount))
    # print(tlabel[maxlabelcount].item())
    tt1=time.time()
    c[c!=torch.mode(c[d])[0].item()]=False
    tt2+=time.time()-tt1
    # c[c!=maxlabel]=False
torch.cuda.synchronize()
print((time.time()-t) * 1000)
print((t1 - t) * 1000)
print(tt2*1000)
print(b[0, :20, :20])
