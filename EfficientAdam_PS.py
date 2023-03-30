from torch.distributed import rpc

from utils import *
from parameter_server import BatchUpdateParameterServer


class EfficientAdamParameterServer(BatchUpdateParameterServer):
    def __init__(self, device, batch_update_size=batch_update_size, num_workers=batch_update_size, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, quantize=quantize, resume_file=''):
        super().__init__(device=device, batch_update_size=batch_update_size, num_workers=num_workers,
                         learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, resume_file=resume_file)
        self.error = [torch.zeros_like(p) for p in self.model.parameters()]
        self.delta_hat = [torch.zeros_like(p) for p in self.model.parameters()]
        self.quantize = quantize

    @torch.no_grad()
    def update_logic(self, fut):
        timed_log(f'PS start update model')
        delta_tilde = [self.quantize(delta_hat /
                                     self.num_workers) + self.error[i] for i, delta_hat in enumerate(self.delta_hat)]
        for i, e in enumerate(self.error):
            e.add_(self.delta_hat[i] - delta_tilde[i])
        for i, p in enumerate(self.model.to(self.device).parameters()):
            p -= delta_tilde[i]
        self.add_bits_curr_epoch(
            sum([delta.nelement() * delta.element_size() for delta in delta_tilde]))
        fut.set_result(delta_tilde)

    def _update_model(self, worker, data):
        fut = self.future_model
        with self.lock:
            timed_log(f'PS got update from trainer{worker+1}')
            for i, delta in enumerate(data['delta']):
                self.delta_hat[i] += delta
            self.curr_update_size += 1
            self.add_comm_curr_epoch()
            self.add_bits_curr_epoch(
                sum([delta.nelement() * delta.element_size() for delta in data['delta']]))
            if self.curr_update_size >= self.batch_update_size:
                self.update_logic(fut)
                self.curr_update_size = 0
                self.future_model = torch.futures.Future()
        return fut
