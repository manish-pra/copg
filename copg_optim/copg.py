import math
import time

import torch
import torch.autograd as autograd

from copg_optim.utils import zero_grad, conjugate_gradient, general_conjugate_gradient #Need for zero grad and conjugate gradient functions

class CoPG(object):
    def __init__(self, max_params, min_params, lr=1e-3, weight_decay=0, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
            # return self.norm_cgx, self.norm_cgy
        else:
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')

    def step(self, ob_tot,lp1,lp2):
        grad_x = autograd.grad(lp1, self.max_params, create_graph=True, retain_graph=True) # can remove create graph
        grad_x_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_x])
        grad_y = autograd.grad(lp2, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_y])
        tot_grad_y = autograd.grad(ob_tot.mean(), self.min_params, create_graph=True, retain_graph=True)
        tot_grad_y = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_y])

        tot_grad_xy = autograd.grad(tot_grad_y, self.max_params, grad_outputs=grad_y_vec, retain_graph=True)
        hvp_x_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_xy]) #tot_xy

        tot_grad_x = autograd.grad(ob_tot.mean(), self.max_params, create_graph=True, retain_graph=True)
        tot_grad_x = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_x])

        tot_grad_yx = autograd.grad(tot_grad_x, self.min_params, grad_outputs=grad_x_vec, retain_graph=True)
        hvp_y_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_yx])

        p_x = torch.add(grad_x_vec, - self.lr * hvp_x_vec)
        p_y = torch.add(grad_y_vec, self.lr * hvp_y_vec)

        if self.collect_info:
            self.norm_px = torch.norm(p_x, p=2)
            self.norm_py = torch.norm(p_y, p=2)
            self.timer = time.time()
        if self.solve_x:
            cg_y, self.iter_num = conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                     tot_grad_x=tot_grad_y, tot_grad_y=tot_grad_x,
                                                     x_params=self.min_params,
                                                     y_params=self.max_params, b=p_y, x=self.old_y,
                                                     nsteps=p_y.shape[0],# // 10000,
                                                     lr=self.lr, device=self.device)

            hcg = autograd.grad(tot_grad_y, self.max_params, grad_outputs=cg_y, retain_graph=False)  # yx
            hcg = torch.cat([g.contiguous().view(-1, 1) for g in hcg])
            cg_x = torch.add(grad_x_vec, - self.lr * hcg)
            self.old_x = cg_x
        else:
            cg_x, self.iter_num = conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                     tot_grad_x=tot_grad_x, tot_grad_y=tot_grad_y,
                                                     x_params=self.max_params,
                                                     y_params=self.min_params, b=p_x, x=self.old_x,
                                                     nsteps=p_x.shape[0],# // 10000,
                                                     lr=self.lr, device=self.device)
            hcg = autograd.grad(tot_grad_x, self.min_params, grad_outputs=cg_x, retain_graph=False)  # yx
            hcg = torch.cat([g.contiguous().view(-1, 1) for g in hcg])
            cg_y = torch.add(grad_y_vec, self.lr * hcg)
            self.old_y = cg_y

        if self.collect_info:
            self.timer = time.time() - self.timer

        index = 0
        for p in self.max_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(self.lr * cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise ValueError('CG size mismatch')
        index = 0
        for p in self.min_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(- self.lr * cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise ValueError('CG size mismatch')

        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)
        self.solve_x = False if self.solve_x else True

class RCoPG(object):
    def __init__(self, max_params, min_params, eps=1e-8, beta2=0.99, lr=1e-3, weight_decay=0, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info
        self.square_avgx = None
        self.square_avgy = None
        self.beta2 = beta2
        self.eps = eps
        self.cg_x = None
        self.cg_y = None
        self.count = 0

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, self.timer,\
                   self.iter_num, self.norm_cgx_cal, self.norm_cgy_cal, self.norm_vx, self.norm_vy, self.norm_mx, self.norm_my
        else:
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')

    def step(self, ob, ob_tot,lp1,lp2):
        self.count += 1
        grad_x = autograd.grad(lp1, self.max_params, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_x])
        grad_y = autograd.grad(lp2, self.min_params, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_y])

        if self.square_avgx is None and self.square_avgy is None:
            self.square_avgx = torch.zeros(grad_x_vec.size(), requires_grad=False,
                                           device=self.device)
            self.square_avgy = torch.zeros(grad_y_vec.size(), requires_grad=False,
                                           device=self.device)
        self.square_avgx.mul_(self.beta2).addcmul_(1 - self.beta2, grad_x_vec.data, grad_x_vec.data)
        self.square_avgy.mul_(self.beta2).addcmul_(1 - self.beta2, grad_y_vec.data, grad_y_vec.data)

        # Initialization bias correction
        bias_correction2 = 1 - self.beta2 ** self.count
        self.v_x = self.square_avgx/bias_correction2
        self.v_y = self.square_avgy / bias_correction2

        lr_x = math.sqrt(bias_correction2) * self.lr / self.square_avgx.sqrt().add(self.eps)
        lr_y = math.sqrt(bias_correction2) * self.lr / self.square_avgy.sqrt().add(self.eps)

        scaled_grad_x = torch.mul(lr_x, grad_x_vec).detach()  # lr_x * grad_x
        scaled_grad_y = torch.mul(lr_y, grad_y_vec).detach()  # lr_y * grad_y

        tot_grad_y = autograd.grad(ob_tot.mean(), self.min_params, create_graph=True, retain_graph=True)
        tot_grad_y = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_y])

        tot_grad_xy = autograd.grad(tot_grad_y, self.max_params, grad_outputs=scaled_grad_y, retain_graph=True)
        hvp_x_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_xy]) # D_xy * lr_y * grad_y

        tot_grad_x = autograd.grad(ob_tot.mean(), self.max_params, create_graph=True, retain_graph=True)
        tot_grad_x = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_x])

        tot_grad_yx = autograd.grad(tot_grad_x, self.min_params, grad_outputs=scaled_grad_x, retain_graph=True)
        hvp_y_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_yx]) # D_yx * lr_x * grad_x)

        p_x = torch.add(grad_x_vec, - hvp_x_vec).detach_()  # grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(grad_y_vec, hvp_y_vec).detach_()  # grad_y + D_yx * lr_x * grad_x

        if self.collect_info:
            self.norm_px = torch.norm(p_x, p=2)
            self.norm_py = torch.norm(p_y, p=2)
            self.timer = time.time()

        if self.solve_x:
            p_y.mul_(lr_y.sqrt())
            cg_y, self.iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                     tot_grad_x=tot_grad_y, tot_grad_y=tot_grad_x,
                                                     x_params=self.min_params,
                                                     y_params=self.max_params, b=p_y, x=self.old_y,
                                                     nsteps=p_y.shape[0],# // 10000,
                                                     lr_x=lr_y, lr_y=lr_x, device=self.device)
            #hcg = Hvp_vec(grad_y_vec, self.max_params, cg_y)
            cg_y.detach_().mul_(- lr_y.sqrt())
            hcg = autograd.grad(tot_grad_y, self.max_params, grad_outputs=cg_y, retain_graph=False)  # yx
            hcg = torch.cat([g.contiguous().view(-1, 1) for g in hcg]).add_(grad_x_vec).detach_()
            # grad_x + D_xy * delta y
            cg_x = hcg.mul(lr_x) # this is basically deltax
            # torch.add(grad_x_vec, - self.lr * hcg)
            self.old_x = hcg.mul(lr_x.sqrt())
        else:
            p_x.mul_(lr_x.sqrt())
            cg_x, self.iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                     tot_grad_x=tot_grad_x, tot_grad_y=tot_grad_y,
                                                     x_params=self.max_params,
                                                     y_params=self.min_params, b=p_x, x=self.old_x,
                                                     nsteps=p_x.shape[0],# // 10000,
                                                     lr_x=lr_x, lr_y=lr_y, device=self.device)
            # cg_x.detach_().mul_(p_x_norm)
            cg_x.detach_().mul_(lr_x.sqrt())  # delta x = lr_x.sqrt() * cg_x
            hcg = autograd.grad(tot_grad_x, self.min_params, grad_outputs=cg_x, retain_graph=False)  # yx
            hcg = torch.cat([g.contiguous().view(-1, 1) for g in hcg]).add_(
                grad_y_vec).detach_()
            # grad_y + D_yx * delta x
            cg_y = hcg.mul(- lr_y)
            # cg_y = torch.add(grad_y_vec, self.lr * hcg)
            self.old_y = hcg.mul(lr_y.sqrt())

        if self.collect_info:
            self.timer = time.time() - self.timer

        index = 0
        for p in self.max_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise ValueError('CG size mismatch')
        index = 0
        for p in self.min_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise ValueError('CG size mismatch')

        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)
            self.norm_cgx_cal = torch.norm(self.square_avgx, p=2)
            self.norm_cgy_cal = torch.norm(self.square_avgy, p=2)
            self.norm_vx = torch.norm(self.v_x, p=2)
            self.norm_vy = torch.norm(self.v_y, p=2)
            self.norm_mx = lr_x.max()
            self.norm_my = lr_y.max()
        self.solve_x = False if self.solve_x else True