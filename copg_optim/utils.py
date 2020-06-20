import torch
import torch.autograd as autograd

def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


def conjugate_gradient(grad_x, grad_y,  tot_grad_x, tot_grad_y, x_params, y_params, b, x=None, nsteps=10, residual_tol=1e-18,
                       lr=1e-3, device=torch.device('cpu')): # not able to parameters
    '''
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b: vec
    :param nsteps: max number of steps
    :param residual_tol:
    :return: A ** -1 * b

    h_1 = D_yx * p
    h_2 = D_xy * D_yx * p
    A = I + lr ** 2 * D_xy * D_yx * p
    '''
    if x is None:
        x = torch.zeros((b.shape[0],1), device=device)
    r = b.clone().detach()
    p = r.clone().detach()
    rdotr = torch.dot(r.view(-1), r.view(-1))
    residual_tol = residual_tol * rdotr
    for itr in range(nsteps):
        # To compute Avp
        h_1 = autograd.grad(tot_grad_x, y_params, grad_outputs=p, retain_graph=True)  # yx
        h_1 = torch.cat([g.contiguous().view(-1, 1) for g in h_1])
        h_2 = autograd.grad(tot_grad_y, x_params, grad_outputs=h_1, retain_graph=True)
        h_2 = torch.cat([g.contiguous().view(-1, 1) for g in h_2])
        Avp_ = p + lr * lr * h_2

        alpha = rdotr / torch.dot(p.view(-1), Avp_.view(-1))
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r.view(-1), r.view(-1))
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        #print(itr)
        if rdotr < residual_tol:
            break
    return x, itr + 1

def general_conjugate_gradient(grad_x, grad_y,tot_grad_x, tot_grad_y, x_params, y_params, b, lr_x, lr_y, x=None, nsteps=10,
                               residual_tol=1e-16,
                               device=torch.device('cpu')):
    '''
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:
    :param x:
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b

    '''
    if x is None:
        x = torch.zeros((b.shape[0],1), device=device)
    if tot_grad_x.shape != b.shape:
        raise RuntimeError('CG: hessian vector product shape mismatch')
    lr_x = lr_x.sqrt()
    r = b.clone().detach()
    p = r.clone().detach()
    rdotr = torch.dot(r.view(-1), r.view(-1))
    residual_tol = residual_tol * rdotr
    for i in range(nsteps):
        # To compute Avp
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
        h_1 = autograd.grad(tot_grad_x, y_params, grad_outputs=lr_x*p, retain_graph=True)  # yx
        h_1 = torch.cat([g.contiguous().view(-1, 1) for g in h_1]).mul_(lr_y)
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True).mul_(lr_y)
        # h_1.mul_(lr_y)
        # lr_y * D_yx * b
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).mul_(lr_x)
        h_2 = autograd.grad(tot_grad_y, x_params, grad_outputs=h_1, retain_graph=True)
        h_2 = torch.cat([g.contiguous().view(-1, 1) for g in h_2]).mul_(lr_x)
        # h_2.mul_(lr_x)
        # lr_x * D_xy * lr_y * D_yx * b
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p.view(-1), Avp_.view(-1))
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r.view(-1), r.view(-1))
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x, i + 1

