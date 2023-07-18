import torch
from torch.utils import cpp_extension
import time
import cuda_ccl
# cuda_module = cpp_extension.load(name="ccl", sources=["CCL_ext.cpp", "CCL.cu"], verbose=True)
a=torch.zeros([20, 1080, 1920], dtype=torch.uint8, device='cuda:0')
b=torch.zeros([20, 1080, 1920], dtype=torch.int32, device='cuda:0')
a[:, 1:5, 5:8]=1
a[:, 10:11, 15:18]=1
torch.cuda.synchronize()
t=time.time()
cuda_ccl.torch_ccl(b,a,a.shape[1],a.shape[2])
# for i in range(b.shape[0]):
#     c=b[i]
    # d=c[c>0]
    # tlabel, tcount = torch.unique(c[c>0], return_counts=True)
    # maxlabelcount = torch.argmax(tcount)
    # maxlabel = tlabel[maxlabelcount].item()
    # print(tlabel, tcount, torch.argmax(tcount))
    # print(tlabel[maxlabelcount].item())
    # c[c!=maxlabel]=0
torch.cuda.synchronize()
print((time.time()-t) * 1000)
# print(b[0, :20, :20])