import torch
from torch import nn
import torch.nn.functional as f
from torch.distributed import rpc

from utils import *
from parameter_server import *


class Trainer(object):
    def __init__(self, ps_rref, worker, device):
        self.ps_rref = ps_rref
        self.loss_fn = nn.CrossEntropyLoss()
        self.model_old = ps_rref.rpc_sync().get_model()
        self.device = device
        self.worker = worker
        self.name = rpc.get_worker_info().name
        timed_log(f'{self.name} request for loader')
        self.trainloader = self.ps_rref.rpc_sync().get_trainloader(worker)
        timed_log(f'{self.name} finished request for loader')

    def train_pre_batch(self, i, model_fresh, inputs, labels):
        timed_log(f"{self.name} processing batch ({i+1})")
        loss = self.loss_fn(model_fresh(inputs), labels)
        loss.backward()
        grad = [p.grad for p in model_fresh.cpu().parameters()]
        return loss, {"grad": grad}

    def train(self, retrieve_model=True):
        model_fresh = self.model_old
        timed_log(f'{self.name} start training')
        timed_log(f'{self.name} dataloader size: {len(self.trainloader)}')
        for i, (inputs, labels) in enumerate(self.trainloader):
            model_fresh.to(self.device)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            _, data = self.train_pre_batch(i, model_fresh, inputs, labels)
            self.train_post_batch(model_fresh=model_fresh, data=data)
            if retrieve_model:
                model_fresh = self.ps_rref.rpc_sync().get_model()

    def train_post_batch(self, model_fresh, data):
        self.thrd = rpc.rpc_sync(
            self.ps_rref.owner(),
            BatchUpdateParameterServer.update_model,
            args=(self.ps_rref, self.worker, data),
        )
        timed_log(f'{self.name} received new thrd')
