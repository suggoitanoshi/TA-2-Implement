from utils import *
from trainer import *

from TA_PS import TAParameterServer


class TATrainer(Trainer):
    def __init__(self, ps_rref, worker, device, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, quantize=quantize, delay_bound=delay_bound, **kwargs):
        super().__init__(ps_rref=ps_rref, worker=worker, device=device, **kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.quantize = quantize
        self.delay = 0
        self.delay_bound = delay_bound
        self.thrd = 0

        if kwargs['data'] is None:
            self.m = [0 for _ in self.model_old.parameters()]
            self.v = [0 for _ in self.model_old.parameters()]
            self.e = [0 for _ in self.model_old.parameters()]
        else:
            self.m = [m.to(self.device) for m in kwargs['data']['m']]
            self.v = [v.to(self.device) for v in kwargs['data']['v']]
            self.e = [e.to(self.device) for e in kwargs['data']['e']]

    def train_pre_batch(self, i, model_fresh, inputs, labels):
        loss, data = super().train_pre_batch(
            i=i, model_fresh=model_fresh, inputs=inputs, labels=labels)
        grad = [grad.to(self.device) for grad in data['grad']]
        self.delay += 1
        delta = []
        if self.delay >= self.delay_bound:
            self.model_old = model_fresh
            self.delay = 1
            for layer, p in enumerate(model_fresh.parameters()):
                p.grad.zero_()
                self.v[layer] = self.beta_2 * \
                    self.v[layer] + \
                    (1 - self.beta_2)*torch.pow(grad[layer], 2)
                vsqrt = torch.sqrt(self.v[layer]).add_(epsilon)
                self.m[layer] = self.beta_1 * self.m[layer] + (
                    1 - self.beta_1)*torch.pow(grad[layer], 1)
                d_raw = self.learning_rate*self.m[layer]/vsqrt + \
                    self.e[layer]
                d = self.quantize(d_raw, device=self.device)
                self.e[layer] += d_raw - d
                delta.append(d.to('cpu'))
        else:
            self.model_old.to(self.device)
            self.loss_fn(self.model_old(inputs), labels).backward()
            old_grad = [p.grad for p in self.model_old.parameters()]
            diff = [torch.norm(g - old_g)**2 for g,
                    old_g in zip(grad, old_grad)]
            diff = sum(diff)
            if diff >= self.thrd:
                timed_log(f'{self.name} reporting delta')
                for layer, p in enumerate(model_fresh.parameters()):
                    p.grad.zero_()
                    self.v[layer] = self.beta_2 * \
                        self.v[layer] + \
                        (1 - self.beta_2)*torch.pow(grad[layer], 2)
                    vsqrt = torch.sqrt(self.v[layer]).add_(epsilon)
                    self.m[layer] = self.beta_1 * self.m[layer] + (
                        1 - self.beta_1)*torch.pow(grad[layer], 1)
                    d_raw = self.learning_rate*self.m[layer]/vsqrt + \
                        self.e[layer]
                    d = self.quantize(d_raw, device=self.device)
                    self.e[layer] += d_raw - d
                    delta.append(d.to('cpu'))
                self.delay = 0
            else:
                delta = None
                timed_log(f'{self.name} skip reporting delta')
        return loss, {"delta": delta}

    def train(self):
        super().train(retrieve_model=False)
        return {'m': [m.to('cpu') for m in self.m], 'v': [v.to('cpu') for v in self.v], 'e': [e.to('cpu') for e in self.e]}

    def train_post_batch(self, model_fresh, data):
        ret = rpc.rpc_sync(
            self.ps_rref.owner(),
            TAParameterServer.update_model,
            args=(self.ps_rref, self.worker, data),
        )
        delta_new = ret['delta_tilde']
        self.thrd = ret['thrd']
        with torch.no_grad():
            for i, p in enumerate(model_fresh.to(self.device).parameters()):
                p.add_(-delta_new[i].to(self.device))
        timed_log(f'{self.name} received new delta')
