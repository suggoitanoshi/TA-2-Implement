from torch.distributed import rpc

from utils import *
from parameter_server import BatchUpdateParameterServer


class TAParameterServer(BatchUpdateParameterServer):
    def __init__(self, device, batch_update_size=batch_update_size, num_workers=batch_update_size, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, quantize=quantize, c=c, dmax=dmax, resume_file='', **kwargs):
        super().__init__(device=device, batch_update_size=batch_update_size, num_workers=num_workers,
                         learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, resume_file=resume_file, **kwargs)
        self.quantize = quantize
        self.thrd_scale = c/dmax

    def _initialize(self):
        super()._initialize()
        self.error = [torch.zeros_like(p).to(self.device)
                      for p in self.model.parameters()]
        self.delta_hat = [None for _ in range(self.num_workers)]
        self.triggerlist = [0 for _ in range(dmax)]

    @torch.no_grad()
    def update_logic(self, fut):
        timed_log(f'PS start update model')
        delta_hat_reduced = [torch.zeros_like(p).to(
            self.device) for p in self.model.parameters()]
        diff = 0
        buf = 0
        for i in range(self.num_workers):
            for dhr, dh in zip(delta_hat_reduced, self.delta_hat[i]):
                dhr += dh

        delta_tilde = [self.quantize(delta_hat.to(self.device) /
                                     self.num_workers, device=self.device) + self.error[i] for i, delta_hat in enumerate(delta_hat_reduced)]
        for i, e in enumerate(self.error):
            buf = 0
            for j in range(self.num_workers):
                buf += self.delta_hat[j][i] / self.num_workers + e * epsilon
            diff += (torch.norm(buf)*self.learning_rate)**2
            e.add_(delta_hat_reduced[i] - delta_tilde[i])
        for i, p in enumerate(self.model.to(self.device).parameters()):
            p.add_(-delta_tilde[i])
        self.triggerlist.append(diff.item())
        self.triggerlist.pop(0)
        thrd = sum(self.triggerlist)
        delta_tilde = [d.to('cpu') for d in delta_tilde]
        self.add_bits_curr_epoch(
            sum([delta.nelement() * delta.element_size() for delta in delta_tilde]))
        self.add_bits_curr_epoch(64)
        fut.set_result({"delta_tilde": delta_tilde, "thrd": thrd})

    def serialize(self):
        return {**super().serialize(), 'error': [e.to('cpu') for e in self.error], 'delta_hat': [[d.to('cpu') for d in delta_hat] for delta_hat in self.delta_hat], 'triggerlist': self.triggerlist}

    def _deserialize(self, data):
        super()._deserialize(data)
        self.error = [e.to(self.device) for e in data['error']]
        self.delta_hat = [[d.to(self.device) for d in delta_hat]
                          for delta_hat in data['delta_hat']]
        self.triggerlist = data['triggerlist']

    def _update_model(self, worker, data):
        fut = self.future_model
        with self.lock:
            timed_log(f'PS got update from trainer{worker+1}')
            if data['delta'] is not None:
                self.delta_hat[worker] = [
                    d.to(self.device) for d in data['delta']]
                self.add_comm_curr_epoch()
                self.add_bits_curr_epoch(
                    sum([delta.nelement() * delta.element_size() for delta in data['delta'] if delta is not None]))
            self.curr_update_size += 1
            if self.curr_update_size >= self.batch_update_size:
                self.update_logic(fut)
                self.curr_update_size = 0
                self.future_model = torch.futures.Future()
        return fut
