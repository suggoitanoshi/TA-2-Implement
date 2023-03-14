from utils import *
from trainer import *

from TA_PS import TAParameterServer


class TATrainer(Trainer):
    def __init__(self, ps_rref, worker, device, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, quantize=quantize, delay_bound=delay_bound):
        super().__init__(ps_rref=ps_rref, worker=worker, device=device)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.quantize = quantize
        self.delay = 0
        self.delay_bound = delay_bound
        self.thrd = 0

        self.momentum_dict = {}
        for layer, _ in enumerate(self.model_old.parameters()):
            self.momentum_dict[f'weight_m_{layer}'] = 0
            self.momentum_dict[f'weight_v_{layer}'] = 0
            self.momentum_dict[f'error_{layer}'] = 0

    def train_pre_batch(self, i, model_fresh, inputs, labels):
        loss, data = super().train_pre_batch(
            i=i, model_fresh=model_fresh, inputs=inputs, labels=labels)
        grad = data['grad']
        self.delay += 1
        delta = []
        if self.delay >= delay_bound:
            self.model_old = model_fresh
            self.delay = 1
        else:
            self.loss_fn(self.model_old(inputs), labels).backward()
            old_grad = [p.grad for p in self.model_old.cpu().parameters()]
            diff = [torch.norm(g - old_g)**2 for g,
                    old_g in zip(grad, old_grad)]
            diff = sum(diff)
            if diff >= self.thrd:
                timed_log(f'{self.name} reporting delta')
                for layer, p in enumerate(model_fresh.cpu().parameters()):
                    v = self.beta_2 * \
                        self.momentum_dict[f'weight_v_{layer}'] + \
                        (1 - self.beta_2)*torch.norm(grad[layer], 2)
                    m = self.beta_1 * \
                        self.momentum_dict[f'weight_m_{layer}'] + \
                        (1 - self.beta_1)*torch.norm(grad[layer], 1)
                    d = self.quantize(self.learning_rate*m /
                                      torch.sqrt(v) + self.momentum_dict[f'error_{layer}'])
                    self.momentum_dict[f'error_{layer}'] = self.learning_rate * m / \
                        torch.sqrt(v) + \
                        self.momentum_dict[f'error_{layer}'] - d
                    delta.append(d)
                self.delay = 0
            else:
                delta = None
                timed_log(f'{self.name} skip reporting delta')
        return loss, {"delta": delta, "grad": grad}

    def train(self):
        return super().train(retrieve_model=False)

    def train_post_batch(self, model_fresh, data):
        delta_new = rpc.rpc_sync(
            self.ps_rref.owner(),
            TAParameterServer.update_model,
            args=(self.ps_rref, self.worker, data),
        )
        with torch.no_grad():
            for i, p in enumerate(model_fresh.cpu().parameters()):
                p.add_(delta_new[i])
        timed_log(f'{self.name} received new delta')
