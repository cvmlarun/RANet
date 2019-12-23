#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:42:16 2019

@author: arun
"""

import torch
import torchvision
from RANet_model import RANet as Net
from RANet_lib.RANet_Model_imagenet import *
import timeit

#model = torchvision.models.resnet18(pretrained=True)
model = Net(with_relu=0, pretrained=False, type='single_object')
model.eval()
example = [torch.rand(1, 3, 480, 864, dtype=torch.float32), torch.zeros(1, 512, 15, 27, dtype=torch.float32), torch.zeros(1, 1, 480, 864, dtype=torch.float32),torch.rand(1, 1, 480, 864, dtype=torch.float32)]
fpath = "/home/arun/RANet/models/RANet_video_single.pth"
checkpoint = torch.load(fpath)
model.state_dict()
model.load_state_dict(checkpoint)
traced_script_module = torch.jit.script(model)
print(traced_script_module.code)
traced_script_module.save("model1.pt")

loaded = torch.jit.load("model1.pt")

base_feature = traced_script_module.forward_first(example[0])
loaded.eval()

Kernel, Feature, base_features1, msk_p, m  = traced_script_module.forward(*example)
Correlation,a = traced_script_module.corr_fun(Kernel, Feature)
outputs, b = traced_script_module.forward_post(Correlation, base_features1, msk_p, m)

start_time = timeit.default_timer()
for i in range(40):
    Kernel, Feature, base_features1, msk_p, m  = traced_script_module.forward(*example)
    Correlation,a = traced_script_module.corr_fun(Kernel, Feature)
    outputs, b = traced_script_module.forward_post(Correlation, base_features1, msk_p, m)
end_time = timeit.default_timer()

print("time =", (end_time - start_time)/40)