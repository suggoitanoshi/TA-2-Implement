from torch.distributed import rpc
from utils import *
from trainer import Trainer
from CADA_PS import CADAParameterServer


class CADATrainer(Trainer):
    def __init__(self, ps_rref, worker, device, delay_bound=delay_bound, **kwargs):
        super().__init__(ps_rref=ps_rref, worker=worker, device=device, **kwargs)
        self.delay = 0
        self.delay_bound = delay_bound
        self.thrd = 0

    def train_pre_batch(self, i, model_fresh, inputs, labels):
        loss, data = super().train_pre_batch(
            i=i, model_fresh=model_fresh, inputs=inputs, labels=labels)
        grad = data['grad']
        self.delay += 1
        if self.delay >= self.delay_bound:
            self.model_old = model_fresh
            self.delay = 1
        else:
            self.model_old.to(self.device)
            self.loss_fn(self.model_old(inputs), labels).backward()
            old_grad = [p.grad for p in self.model_old.cpu().parameters()]
            diff = [torch.norm(g - old_g)**2 for g,
                    old_g in zip(grad, old_grad)]
            diff = sum(diff)
            if diff >= self.thrd:
                timed_log(f'{self.name} reporting grads')
                self.model_old = model_fresh
                self.delay = 0
            else:
                timed_log(f'{self.name} skip reporting grads')
                grad = None
        [p.grad.zero_() for p in self.model_old.cpu().parameters()]
        return loss, {"grad": grad}

    def train(self, **kwargs):
        return {'data': None, **super().train(kwargs)}

    def train_post_batch(self, model_fresh, data):
        self.thrd = rpc.rpc_sync(
            self.ps_rref.owner(),
            CADAParameterServer.update_model,
            args=(self.ps_rref, self.worker, data),
        )
        timed_log(f'{self.name} received new thrd')
