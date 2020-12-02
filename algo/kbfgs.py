import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import AddBias

from algo.kfac import compute_cov_a

# TODO: In order to make this code faster:
# 1) Implement _extract_patches as a single cuda kernel
# 2) Compute QR decomposition in a separate process
# 3) Actually make a general KFAC optimizer so it fits PyTorch


def double_damping(s, y, H, damping):
    # DD_v2
    
    
    # first step
    
    mu_1 = 0.2
    
    s_T_y = torch.dot(s, y)
    
    Hy = torch.mv(H ,y)
    
    yHy = torch.dot(y, Hy)
    
    sy_over_yHy_before = s_T_y.item() / yHy.item()
    
#     if sy_over_yHy_before > alpha:
#         theta = 1
#         damping_status = 0
#         1
#     else:
        
    if sy_over_yHy_before <= mu_1:
        theta =  ((1-mu_1) * yHy / (yHy - s_T_y)).item()

#         original_s_l_a = s_l_a

        s = theta * s + (1-theta) * Hy

#         damping_status = 1
    
    # second step
    
#     alpla = math.sqrt(damping)
    mu_2 = damping
    
    y = y + mu_2 * s
    
    return s, y

def BFGS_update(H, s, y):
    
    rho_inv = torch.dot(s, y).item()
    
    if math.isnan(rho_inv):
        print('rho_inv is nan')
        sys.exit()
    
#     if not (rho_inv > 0):
#         print('rho_inv')
#         print(rho_inv)
#         sys.exit()
    
    assert rho_inv > 0
    
    rho = 1 / rho_inv
    
    Hy = torch.mv(H, y)
    H_new = H.data +\
    (rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s) -\
    rho * (torch.ger(s, Hy) + torch.ger(Hy, s))
    
    return H_new
    
    
    
    


def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def compute_mean_a(a, classname, layer_info, fast_cnn, if_homo):
#     batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            
            print('need to check')
            sys.exit()
            
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            
#             print('a.size()')
#             print(a.size())
            
            a = _extract_patches(a, *layer_info)
        
            mean_a = a.mean(dim=(0,1,2))
            
#             return mean_a
            
#             a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
            
            
#             print('a.size()')
#             print(a.size())
    elif classname == 'AddBias':
        
        print('should not reach here')
        sys.exit()
        
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()
            
    else:
        
#         print('a.size()')
#         print(a.size())
            
            
#         print('classname')
#         print(classname)
            
            
        mean_a = a.mean(dim=0)
    
#     print('mean_a.size()')
#     print(mean_a.size())
    
    if if_homo:
        is_cuda = a.is_cuda
        homo_one = torch.ones(1)
        if is_cuda:
            homo_one = homo_one.cuda()

#         print('homo_one.size()')
#         print(homo_one.size())
        
        mean_a = torch.cat((mean_a, homo_one), dim=0)
        
#         print('mean_a.size()')
#         print(mean_a.size())
    
    
    
    
    
#     sys.exit()

#     return a.t() @ (a / batch_size)
    return mean_a


# def compute_cov_a(a, classname, layer_info, fast_cnn):
#     batch_size = a.size(0)

#     if classname == 'Conv2d':
#         if fast_cnn:
#             a = _extract_patches(a, *layer_info)
#             a = a.view(a.size(0), -1, a.size(-1))
#             a = a.mean(1)
#         else:
#             a = _extract_patches(a, *layer_info)
#             a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
#     elif classname == 'AddBias':
#         is_cuda = a.is_cuda
#         a = torch.ones(a.size(0), 1)
#         if is_cuda:
#             a = a.cuda()

#     return a.t() @ (a / batch_size)


# def compute_mean_g(g, classname, layer_info, fast_cnn):
def compute_mean_g(g, classname, fast_cnn):
    batch_size = g.size(0)
    
    if classname == 'Conv2d':
        if fast_cnn:
            
            print('need to check')
            
            sys.exit()
            
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            return g.mean(dim=(0,2,3))
    elif classname == 'AddBias':
        print('should not reach here')
        sys.exit()
        
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size

    return g_.mean(dim=0)

def compute_mean_h(h, classname, fast_cnn):
#     batch_size = h.size(0)
    
    if classname == 'Conv2d':
        if fast_cnn:
            
            print('need to check')
            
            sys.exit()
            
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            
#             print('h.size()')
#             print(h.size())
            
            return h.mean(dim=(0,2,3))
    elif classname == 'AddBias':
        print('should not reach here')
        sys.exit()
        
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    return h.mean(dim=0)


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


class KBFGSOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.25,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
#                  damping=1e-2,
                 damping=1e-1,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10,
                 if_homo=False):
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

#         super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        super(KBFGSOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}
        
        self.kbfgs_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        
        self.H_A = {}
        self.H_G = {}
        
        self.mean_a = {}
        self.h_G_cur = {}
        self.h_G_next = {}
        self.g_G_cur = {}
        self.g_G_next = {}

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
        
#         print('self.momentum')
#         print(self.momentum)

        self.optim = optim.SGD(
            model.parameters(),
            lr=self.lr * (1 - self.momentum),
            momentum=self.momentum)

    def _save_input(self, module, input):
        
#         print('torch.is_grad_enabled()')
#         print(torch.is_grad_enabled())
        
        
        
        # When sample, torch.no_grad is used
        # When optimizing, only one forward pass for kfac
#         if torch.is_grad_enabled() and self.steps % self.Ts == 0:
        if torch.is_grad_enabled() and self.steps % self.Ts == 0 and self.kbfgs_stats_cur:
            
            classname = module.__class__.__name__
        
#             print('classname in _save_input')
#             print(classname)
        
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
            
            mean_a = compute_mean_a(input[0].data, classname, layer_info, self.fast_cnn, self.if_homo)
            
            self.mean_a[module] = mean_a.clone()
            

    def _save_output(self, module, input, output):
        
        if torch.is_grad_enabled() and (self.kbfgs_stats_cur or self.kbfgs_stats_next):
        
            classname = module.__class__.__name__
            
#             print('output.size()')
#             print(output.size())
            
            
#             mean_h = compute_mean_h(output[0].data, classname, self.fast_cnn)
            mean_h = compute_mean_h(output.data, classname, self.fast_cnn)
            
            if self.kbfgs_stats_cur:
                
                if self.steps == 0:
                    self.h_G_cur[module] = mean_h.clone()
                    
                update_running_stat(mean_h, self.h_G_cur[module], self.stat_decay)
            elif self.kbfgs_stats_next:
                
#                 if self.steps == 0:
                if self.steps == 1:
                    self.h_G_next[module] = mean_h.clone()
                    
                update_running_stat(mean_h, self.h_G_next[module], self.stat_decay)
            else:
                print('should not reach here')
                sys.exit()

    def _save_grad_output(self, module, grad_input, grad_output):
#         if self.acc_stats:
#         if self.kbfgs_stats:
#         if 1:
        if self.kbfgs_stats_cur or self.kbfgs_stats_next:
        
            classname = module.__class__.__name__
#             layer_info = None
#             if classname == 'Conv2d':
#                 layer_info = (module.kernel_size, module.stride,
#                               module.padding)

#             gg = compute_cov_g(grad_output[0].data, classname, layer_info,
#                                self.fast_cnn)
            
            

            # Initialize buffers
#             if self.steps == 0:
#                 self.m_gg[module] = gg.clone()

#             update_running_stat(gg, self.m_gg[module], self.stat_decay)
            
            
            mean_g = compute_mean_g(grad_output[0].data, classname, self.fast_cnn)
            
            if self.kbfgs_stats_cur:
                
                if self.steps == 0:
                    self.g_G_cur[module] = mean_g.clone()
                    
                update_running_stat(mean_g, self.g_G_cur[module], self.stat_decay)
            elif self.kbfgs_stats_next:
                
#                 if self.steps == 0:
                if self.steps == 1:
                    self.g_G_next[module] = mean_g.clone()
                    
                update_running_stat(mean_g, self.g_G_next[module], self.stat_decay)
            else:
                print('should not reach here')
                sys.exit()

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            
            if classname in self.known_modules:
#             if classname in self.kbfgs_modules:

#                 assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
#                                     "You must have a bias as a separate layer"

                self.modules.append(module)
        
                if classname in self.kbfgs_modules:
                    module.register_forward_pre_hook(self._save_input)
                    module.register_forward_hook(self._save_output)
                    module.register_backward_hook(self._save_grad_output)
                    
    
                    
                    
    def post_step(self):
        
        for i, m in enumerate(self.modules):
#             assert len(list(m.parameters())
#                        ) == 1, "Can handle only one parameter at the moment"
            
            classname = m.__class__.__name__
            
            
            
            if classname == 'AddBias':
                continue
                
            la = self.damping + self.weight_decay
            
            
            # compute BFGS for G here
            
            # need s, y
            
            # for G
            
            # compute s, y
            
#             print('self.h_G_cur[m].size()')
#             print(self.h_G_cur[m].size())
            
#             print('self.h_G_next[m].size()')
#             print(self.h_G_next[m].size())
            
            s_G = self.h_G_next[m] - self.h_G_cur[m]
            
            y_G = self.g_G_next[m] - self.g_G_cur[m]
        
            s_G, y_G = double_damping(s_G.data, y_G.data, self.H_G[m].data, math.sqrt(la))
            
#             print('torch.dot(s_G, s_G) / torch.dot(s_G, y_G)')
#             print(torch.dot(s_G, s_G) / torch.dot(s_G, y_G))
            
#             print('1 / math.sqrt(la)')
#             print(1 / math.sqrt(la))
            
#             assert torch.dot(s_G, s_G) / torch.dot(s_G, y_G) <= 1 / math.sqrt(la)

            self.H_G[m] = BFGS_update(self.H_G[m].data, s_G.data, y_G.data)
            
            
            
#             sys.exit()

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
            
            p = next(m.parameters())
            
            if self.if_homo:
                assert len(list(m.parameters())) == 2
                p_bias = list(m.parameters())[1]

            la = self.damping + self.weight_decay
            
            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data
                
            if self.if_homo:
                p_grad_mat = torch.cat(
                    (p_grad_mat, p_bias.grad.data.unsqueeze(1)), dim=1
                )
            
            if classname == 'AddBias':
                
                print('need to check if homo')
                
                sys.exit()
                
#                 print('use sgd')
                
#                 updates[p] = p.grad.data / la
                updates[p] = p.grad.data
                continue
        
        
            # initialize H
            
            if self.steps == 0:
                self.H_A[m] = torch.eye(self.m_aa[m].size(0))
                self.H_G[m] = torch.eye(self.g_G_cur[m].size(0))
                
                is_cuda = self.m_aa[m].is_cuda
                
                if is_cuda:
                    self.H_A[m] = self.H_A[m].cuda()
                    self.H_G[m] = self.H_G[m].cuda()
                
                
                
                
                
                
#                 print('need cuda?')
            
            # compute BFGS for A here
            
            
            
            # need s, y
            
            # for A
            
            # compute s

#             print('self.H_A[m].size()')
#             print(self.H_A[m].size())
        
#             print('self.mean_a[m].size()')
#             print(self.mean_a[m].size())
            
#             s_A = self.s_A[m]
            s_A = torch.mv(self.H_A[m], self.mean_a[m])
            
            # compute y
            
            y_A = torch.mv(self.m_aa[m], s_A) + math.sqrt(la) * s_A
            
            # compute H_A
            self.H_A[m] = BFGS_update(self.H_A[m].data, s_A.data, y_A.data)
        
            
            
            

            if 0:
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                
                # use eigen decompostion to compute inverse
                # damping is included when computing direction
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg[m], eigenvectors=True)

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())
                
                
            
#             print('p_grad_mat.size()')
#             print(p_grad_mat.size())
            
                
#             v1 = self.H_G[m].t() @ p_grad_mat @ self.H_A[m]
            v = self.H_G[m].t() @ p_grad_mat @ self.H_A[m]

            # @ is matmul
#             v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            
#             v2 = v1 / (
#                 self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
#             v = self.Q_g[m] @ v2 @ self.Q_a[m].t()


#             print('p.size()')
#             print(p.size())
        
#             print('p_bias.size()')
#             print(p_bias.size())
            
#             print('v.size()')
#             print(v.size())
            
            if self.if_homo:
                v_bias = v[:, -1]
                v = v[:, :-1]

            v = v.view(p.grad.data.size())
            updates[p] = v
            
            if self.if_homo:
                updates[p_bias] = v_bias
        
            
            
#         sys.exit()

        vg_sum = 0
        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        # this is the SGD step
        self.optim.step()
        
        
        
        self.steps += 1
        
        
