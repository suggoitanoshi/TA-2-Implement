import threading
import random

import torch
from torch.distributed import rpc
from torch.utils.data import DataLoader, SubsetRandomSampler

from resnet import resnet20
from fashion_mnist_model import FashionMnistNet
import torchvision
from torchvision import transforms
import torch.nn.functional as f

from utils import *


class BatchUpdateParameterServer(object):
    def __init__(self, device, batch_update_size=batch_update_size, num_workers=batch_update_size, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, resume_file=''):
        self.model = FashionMnistNet()
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.lock = threading.Lock()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.future_model = torch.futures.Future()
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.first = True
        self.comm_current_epoch = 0
        self.bits_current_epoch = 0
        self.device = device

        if resume_file != '':
            self._deserialize(torch.load(resume_file))
        else:
            self._initialize()

        self.momentum_dict = {}
        for layer, _ in enumerate(self.model.parameters()):
            self.momentum_dict[f'weight_q_{layer}'] = 0
            self.momentum_dict[f'weight_v_{layer}'] = 0
            self.momentum_dict[f'weight_v_hat_{layer}'] = 0

        self.trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                          download=True,
                                                          transform=transform_train)
        self.sorted_idx = random.shuffle(
            sort_idx(self.trainset, num_classes, 50000))
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True,
                                                    transform=transform_test)

        self.testloader = DataLoader(testset)
        self.trainloader = [DataLoader(self.trainset, sampler=SubsetRandomSampler(range(i*nsample, (i+1)*nsample)), batch_size=batch_size) for i in range(self.num_workers)]

    def set_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def get_model(self):
        self.add_bits_curr_epoch(sum(
            [p.nelement() * p.element_size() for p in self.model.parameters()]))
        return self.model

    def get_trainloader(self, i):
        return self.trainloader[i]

    def get_testloader(self):
        return self.testloader

    def add_comm_curr_epoch(self):
        self.comm_current_epoch += 1

    def add_bits_curr_epoch(self, bits):
        self.bits_current_epoch += bits

    def get_stats(self):
        return {"comms": self.comm_current_epoch, "bits": self.bits_current_epoch}

    def reset_stats(self):
        self.comm_current_epoch = 0
        self.bits_current_epoch = 0

    def update_adam_params(self):
        diff = 0
        epsilon = 1e-8
        with torch.no_grad():
            for layer, param in enumerate(self.model.parameters()):
                buf = 0
                for i in range(self.num_workers):
                    buf += self.grad[i][layer]
                diff += (torch.norm(buf)*self.learning_rate)**2
                self.momentum_dict[f'weight_q_{layer}'] = self.beta_1 * \
                    self.momentum_dict[f'weight_q_{layer}'] + \
                    (1-self.beta_1) * buf
                self.momentum_dict[f'weight_v_{layer}'] = self.beta_2 * \
                    self.momentum_dict[f'weight_v_{layer}'] + \
                    (1 - self.beta_2) * (buf**2)
                if self.first:
                    self.momentum_dict[f'weight_v_hat_{layer}'] = self.momentum_dict[f'weight_v_{layer}']
                self.momentum_dict[f'weight_v_hat_{layer}'] = torch.max(
                    self.momentum_dict[f'weight_v_hat_{layer}'], self.momentum_dict[f'weight_v_{layer}'])
                param.add_((self.momentum_dict[f'weight_q_{layer}'] / torch.sqrt(
                    self.momentum_dict[f'weight_v_hat_{layer}'] + epsilon)), alpha=-self.learning_rate)
        return diff

    def update_logic(self, fut):
        for g in self.grad:
            if g == None:
                continue
            for p, grad in zip(self.model.parameters(), g):
                p.grad += grad

        for p in self.model.parameters():
            p.grad /= self.num_workers
        diff = self.update_adam_params()
        self.first = False
        return diff

    @ staticmethod
    @ rpc.functions.async_execution
    def update_model(ps_rref, worker, data):
        self = ps_rref.local_value()
        timed_log(
            f"PS got {self.curr_update_size+1}/{self.num_workers} updates")
        return self._update_model(worker, data)

    def _update_model(self, worker, data):
        fut = self.future_model
        return fut

    def serialize(self):
        return {"model": self.model.to('cpu').state_dict()}

    def _deserialize(self, data):
        self.model.load_state_dict(data['model'])

    def _initialize(self):
        pass

    def eval(self):
        timed_log(f'start evaluating model')
        with torch.no_grad():
            loss = 0
            correct = 0
            self.model.to(self.device)
            for input, target in self.testloader:
                input = input.to(self.device)
                target = target.to(self.device)
                output = self.model(input)
                loss += f.cross_entropy(output,
                                        target, reduction='sum').detach().item()
                _, pred = torch.max(output.data, 1)
                correct += (pred == target).sum().item()
            self.model.to('cpu')
        loss /= len(self.testloader.dataset)
        acc = 100. * correct / len(self.testloader.dataset)
        return {"loss": loss, "acc": acc}
