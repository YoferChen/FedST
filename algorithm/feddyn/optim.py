from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer, required

class FedDyn(Optimizer):

    def __init__(self, params,lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0, dyn_alpha=0, alpha=0, eps = 1e-5, centered = False):
        
        self.itr = 0
        self.a_sum = 0
        self.dyn_alpha = dyn_alpha
        self.alpha=alpha
        self.eps=eps
        self.centered = centered
        self.gt_avg = {}
        self.vt = {}


        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedDyn, self).__init__(params, defaults)
        a = 1

    def clean_state(self):
        self.state = defaultdict(dict)

    def __setstate__(self, state):
        super(FedDyn, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    def step(self, local_gradient, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for idx,(p,l_g) in enumerate(zip(group['params'], local_gradient)):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                
                d_p.add_(local_gradient[l_g])
                d_p.add_(self.dyn_alpha, p.data - param_state['old_init'])

                if self.alpha !=0:
                    self.vt[idx] =self.alpha*self.vt.get(idx,0)+(1-self.alpha)*d_p**2
                    vt = self.vt[idx]
                    if self.centered:
                        self.gt_avg[idx] = self.alpha*self.gt_avg.get(idx,0)+(1-self.alpha)*d_p
                        vt = vt - self.gt_avg[p]**2
                    vt = torch.sqrt(vt)+self.eps
                else:
                    vt = torch.ones_like(d_p)

                # apply proximal update
                #d_p.add_(self.mu, p.data - param_state['old_init'])
                p.data += -group['lr']/vt*d_p

        return loss



