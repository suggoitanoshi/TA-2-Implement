from torch.distributed import rpc

from utils import *
from parameter_server import BatchUpdateParameterServer


class TAParameterServer(BatchUpdateParameterServer):
    def __init__(self, device, batch_update_size=batch_update_size, num_workers=batch_update_size, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, quantize=quantize, c=c, dmax=dmax, resume_file=''):
        super().__init__(device=device, batch_update_size=batch_update_size, num_workers=num_workers,
                         learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, resume_file=resume_file)
        self.error = [torch.zeros_like(p).to(self.device)
                      for p in self.model.parameters()]
        self.delta_hat = [torch.zeros_like(p).to(
            self.device) for p in self.model.parameters()]
        self.quantize = quantize
        self.triggerlist = [0 for _ in range(dmax)]
        self.thrd_scale = c/dmax

    @torch.no_grad()
    def update_logic(self, fut):
        timed_log(f'PS start update model')
        delta_tilde = [self.quantize(delta_hat.to(self.device) /
                                     self.num_workers, device=self.device) + self.error[i] for i, delta_hat in enumerate(self.delta_hat)]
        for i, e in enumerate(self.error):
            e.add_(self.delta_hat[i] - delta_tilde[i])
        diff = 0
        for i, p in enumerate(self.model.to(self.device).parameters()):
            p.add_(-delta_tilde[i])
            diff += (torch.norm(self.delta_hat[i] + self.error[i]) *
                     self.learning_rate)**2*self.thrd_scale
        self.triggerlist.append(diff)
        self.triggerlist.pop(0)
        thrd = sum(self.triggerlist)
        delta_tilde = [d.to('cpu') for d in delta_tilde]
        self.add_bits_curr_epoch(
            sum([delta.nelement() * delta.element_size() for delta in delta_tilde]))
        self.add_bits_curr_epoch(64)
        self.delta_hat = [delta_hat.zero_() for delta_hat in self.delta_hat]
        fut.set_result({"delta_tilde": delta_tilde, "thrd": thrd.item()})

    def _update_model(self, worker, data):
        fut = self.future_model
        with self.lock:
            timed_log(f'PS got update from trainer{worker+1}')
            if data['delta'] is not None:
                for i, delta in enumerate(data['delta']):
                    self.delta_hat[i] += delta.to(self.device)
                self.add_comm_curr_epoch()
                self.add_bits_curr_epoch(
                    sum([delta.nelement() * delta.element_size() for delta in data['delta'] if delta is not None]))
            self.curr_update_size += 1
            if self.curr_update_size >= self.batch_update_size:
                self.update_logic(fut)
                self.curr_update_size = 0
                self.future_model = torch.futures.Future()
        return fut
