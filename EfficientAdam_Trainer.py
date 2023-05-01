from torch.distributed import rpc

from utils import *
from trainer import Trainer
from EfficientAdam_PS import EfficientAdamParameterServer


class EfficientAdamTrainer(Trainer):
    def __init__(self, ps_rref, worker, device, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, quantize=quantize, **kwargs):
        super().__init__(ps_rref=ps_rref, worker=worker, device=device, **kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.quantize = quantize

        if kwargs['data'] is None:
            self.v = [0 for _ in self.model_old.parameters()]
            self.m = [0 for _ in self.model_old.parameters()]
            self.e = [0 for _ in self.model_old.parameters()]
        else:
            self.v = [v.to(self.device) for v in kwargs['data']['v']]
            self.m = [m.to(self.device) for m in kwargs['data']['m']]
            self.e = [e.to(self.device) for e in kwargs['data']['e']]

    def train_pre_batch(self, i, model_fresh, inputs, labels):
        loss, data = super().train_pre_batch(
            i=i, model_fresh=model_fresh, inputs=inputs, labels=labels)
        grad = [grad.to(self.device) for grad in data['grad']]
        delta = []
        with torch.no_grad():
            for layer, p in enumerate(model_fresh.parameters()):
                p.grad.zero_()
                self.v[layer] = self.beta_2 * \
                    self.v[layer] + \
                    (1 - self.beta_2)*torch.pow(grad[layer], 2)
                self.m[layer] = self.beta_1 * \
                    self.m[layer] + \
                    (1 - self.beta_1)*torch.pow(grad[layer], 1)
                vsqrt = torch.sqrt(self.v[layer]).add_(epsilon)
                error = self.e[layer]
                d_raw = self.m[layer] * self.learning_rate / vsqrt + error
                d = self.quantize(d_raw, device=self.device)
                self.e[layer] += (d_raw - d)
                delta.append(d.to('cpu'))
        return loss, {"delta": delta}

    def train(self):
        super().train(retrieve_model=False)
        return {'v': [v.to('cpu') for v in self.v], 'm': [m.to('cpu') for m in self.m], 'e': [e.to('cpu') for e in self.e]}

    def train_post_batch(self, model_fresh, data):
        delta_new = rpc.rpc_sync(
            self.ps_rref.owner(),
            EfficientAdamParameterServer.update_model,
            args=(self.ps_rref, self.worker, data),
        )
        with torch.no_grad():
            for i, p in enumerate(model_fresh.to(self.device).parameters()):
                p -= delta_new[i].to(self.device)
        timed_log(f'{self.name} received new delta')
