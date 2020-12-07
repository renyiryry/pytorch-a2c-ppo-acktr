import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import AddBias

# TODO: In order to make this code faster:
# 1) Implement _extract_patches as a single cuda kernel
# 2) Compute QR decomposition in a separate process
# 3) Actually make a general KFAC optimizer so it fits PyTorch


def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    
#     print('x.size()')
#     print(x.size())
    
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def compute_cov_a(a, classname, layer_info, fast_cnn, if_homo):
    batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            
            print('need to check for homo')
            sys.exit()
            
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            
#             print('a.size() after extract')
#             print(a.size())
            
            if if_homo:
                homo_ones = torch.ones(a.size(0), a.size(1), a.size(2), 1)
                is_cuda = a.is_cuda
                if is_cuda:
                    homo_ones = homo_ones.cuda()
                    
                a = torch.cat((a, homo_ones), dim=3)
                
            
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
        
    elif classname == 'AddBias':
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()
            
    else:
        
#         print('classname')
#         print(classname)
        
#         print('a.size()')
#         print(a.size())
        
        if if_homo:
            homo_ones = torch.ones(a.size(0), 1)
            is_cuda = a.is_cuda
            if is_cuda:
                homo_ones = homo_ones.cuda()

            a = torch.cat((a, homo_ones), dim=1)
            
#         print('a.size()')
#         print(a.size())

    return a.t() @ (a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return g_.t() @ (g_ / g.size(0))


def update_running_stat(aa, m_aa, momentum):
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= (1 - momentum)


class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.25,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10,
                 if_homo=False,
                 if_eigen=True):
        defaults = dict()
        
        print('model')
        print(model)

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        if not if_homo:
            split_bias(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}
        
        if if_homo:
            self.kfac_modules = {'Linear', 'Conv2d'}
        else:
            self.kfac_modules = {'Linear', 'Conv2d', 'AddBias'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        
        if if_eigen:
            self.Q_a, self.Q_g = {}, {}
            self.d_a, self.d_g = {}, {}
        else:
            self.H_g, self.H_a = {}, {} 

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf
        
        self.if_homo = if_homo
        self.if_eigen = if_eigen

        self.optim = optim.SGD(
            model.parameters(),
            lr=self.lr * (1 - self.momentum),
            momentum=self.momentum)

    def _save_input(self, module, input):
        
#         print('torch.is_grad_enabled()')
#         print(torch.is_grad_enabled())
        
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            aa = compute_cov_a(input[0].data, classname, layer_info,
                               self.fast_cnn, self.if_homo)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            gg = compute_cov_g(grad_output[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
#                 assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
#                                     "You must have a bias as a separate layer"

                self.modules.append(module)
    
                if classname in self.kfac_modules:
                    module.register_forward_pre_hook(self._save_input)
                    module.register_backward_hook(self._save_grad_output)

    def step(self):
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}
        for i, m in enumerate(self.modules):
#             assert len(list(m.parameters())
#                        ) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
    
#             print('len(list(m.parameters())')
#             print(len(list(m.parameters())))
    
    
            p = next(m.parameters())
        
#             print('p.size()')
#             print(p.size())
            
            if self.if_homo:

                assert len(list(m.parameters())) == 2
                p_bias = list(m.parameters())[1]
                
            
            
#             sys.exit()

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                
                
                if self.if_eigen:
                
                    self.d_a[m], self.Q_a[m] = torch.symeig(
                        self.m_aa[m], eigenvectors=True)
                    self.d_g[m], self.Q_g[m] = torch.symeig(
                        self.m_gg[m], eigenvectors=True)

                    self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                    self.d_g[m].mul_((self.d_g[m] > 1e-6).float())
                    
                else:
                    
                    damping_a = math.sqrt(la)
                    damping_g = math.sqrt(la)
                    
                    is_cuda = self.m_aa[m].is_cuda
                    
                    if is_cuda:
                        m_aa_damped = self.m_aa[m] + damping_a * torch.eye(self.m_aa[m].size(0)).cuda()
                        m_gg_damped = self.m_gg[m] + damping_g * torch.eye(self.m_gg[m].size(0)).cuda()
                    else:
                        m_aa_damped = self.m_aa[m] + damping_a * torch.eye(self.m_aa[m].size(0))
                        m_gg_damped = self.m_gg[m] + damping_g * torch.eye(self.m_gg[m].size(0))
                    
                    self.H_a[m] = torch.inverse(m_aa_damped)
                    self.H_g[m] = torch.inverse(m_gg_damped)
                    
#                     sys.exit()
            


            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data
            
#             print('p_bias.grad.data.unsqueeze(1).size()')
#             print(p_bias.grad.data.unsqueeze(1).size())
            
            if self.if_homo:
                p_grad_mat = torch.cat(
                    (p_grad_mat, p_bias.grad.data.unsqueeze(1)), dim=1
                )
                

            if self.if_eigen:

                v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
                v2 = v1 / (
                    self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
                v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
                
            else:
                v = self.H_g[m].t() @ p_grad_mat @ self.H_a[m]
            

            
            if self.if_homo:
                v_bias = v[:, -1]
                
                v = v[:, :-1]
                
#                 print('v.size()')
#                 print(v.size())
                
#                 print('v_bias.size()')
#                 print(v_bias.size())

            v = v.view(p.grad.data.size())
            updates[p] = v
            
            if self.if_homo:
                updates[p_bias] = v_bias

        vg_sum = 0
        for p in self.model.parameters():
            
#             print('p')
#             print(p)
            
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        self.optim.step()
        self.steps += 1
