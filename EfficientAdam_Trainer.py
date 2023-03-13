from torch.distributed import rpc

from utils import *
from trainer import Trainer
from EfficientAdam_PS import EfficientAdamParameterServer


class EfficientAdamTrainer(Trainer):
    def __init__(self, ps_rref, worker, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, quantize=quantize):
        super().__init__(ps_rref=ps_rref, worker=worker)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.quantize = quantize

        self.momentum_dict = {}
        for layer, _ in enumerate(self.model_old.parameters()):
            self.momentum_dict[f'weight_m_{layer}'] = 0
            self.momentum_dict[f'weight_v_{layer}'] = 0
            self.momentum_dict[f'error_{layer}'] = 0

    def train_pre_batch(self, i, model_fresh, inputs, labels):
        loss, grad = super().train_pre_batch(
            i=i, model_fresh=model_fresh, inputs=inputs, labels=labels)
        delta = []
        for layer, p in enumerate(self.model_old.parameters()):
            v = self.beta_2 * \
                self.momentum_dict[f'weight_v_{layer}'] + \
                (1 - self.beta_2)*torch.norm(grad[layer], 2)
            m = self.beta_1 * \
                self.momentum_dict[f'weight_m_{layer}'] + \
                (1 - self.beta_1)*torch.norm(grad[layer], 1)
            d = self.quantize(self.learning_rate*m /
                              torch.sqrt(v) + self.momentum_dict[f'error_{layer}'])
            self.momentum_dict[f'error_{layer}'] = self.learning_rate * m / \
                torch.sqrt(v) + self.momentum_dict[f'error_{layer}'] - d
            delta.append(d)
        return loss, {"delta": delta}

    def train(self):
        return super().train(retrieve_model=False)

    def train_post_batch(self, delta):
        delta_new = rpc.rpc_sync(
            self.ps_rref.owner(),
            EfficientAdamParameterServer.update_model,
            args=(self.ps_rref, self.worker, delta),
        )
        with torch.no_grad():
            for i, p in enumerate(self.model_old.parameters()):
                p.add_(delta_new[i])
        timed_log(f'{self.name} received new delta')
