get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")
get_ipython().run_line_magic("load_ext", " tensorboard")


import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

import pathlib

torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


tensorboard --logdir=tensorboard


path = pathlib.Path.cwd() / 'tensorboard'
writer = SummaryWriter(path)

x = torch.randn(10, 100)
net = Net()


class Hooks:
    def __init__(self, model):
        self.model = model
        
    def forward_hooks(self):
        for module in net.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.register_forward_hook(self.activation_hook)
                
    def activation_hook(self, module, inp, out):
        writer.add_histogram(f'Pre-Activations/{repr(module)}', out)
        
    def gradient_hooks(self):
        for module in net.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                print(repr(module))
                module.weight.register_hook(self.grad_hook_wrapper(module))
    
    def grad_hook_wrapper(self, module):
        def grad_hook(grad):
            writer.add_histogram(f'Gradients/{repr(module)}', grad)
        return grad_hook


hooks = Hooks(net)
hooks.forward_hooks()
hooks.gradient_hooks()


y = net(x)
y.sum().backward()


net.fc1.weight.register_hook


net.



