from torch.distributed import rpc
from utils import *
from trainer import Trainer
from CADA_PS import CADAParameterServer


class CADATrainer(Trainer):
    def __init__(self, ps_rref, worker, delay_bound=delay_bound):
        super().__init__(ps_rref=ps_rref, worker=worker)
        self.delay = 0
        self.delay_bound = delay_bound
        self.thrd = 0

    def train_pre_batch(self, i, model_fresh, inputs, labels):
        loss, grad = super().train_pre_batch(
            i=i, model_fresh=model_fresh, inputs=inputs, labels=labels)
        self.delay += 1
        if self.delay >= self.delay_bound:
            self.model_old = model_fresh
            self.delay = 1
        else:
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
        return loss, {"grad": grad}

    def train_post_batch(self, grad):
        self.thrd = rpc.rpc_sync(
            self.ps_rref.owner(),
            CADAParameterServer.update_model,
            args=(self.ps_rref, self.worker, grad),
        )
        timed_log(f'{self.name} received new thrd')
