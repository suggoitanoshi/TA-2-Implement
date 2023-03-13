from torch.distributed import rpc

from utils import *
from parameter_server import BatchUpdateParameterServer


class EfficientAdamParameterServer(BatchUpdateParameterServer):
    def __init__(self, batch_update_size=batch_update_size, num_workers=batch_update_size, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, quantize=quantize):
        super().__init__(batch_update_size=batch_update_size, num_workers=num_workers,
                         learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.error = [torch.zeros_like(p) for p in self.model.parameters()]
        self.delta_hat = [torch.zeros_like(p) for p in self.model.parameters()]
        self.quantize = quantize

    @torch.no_grad()
    def update_logic(self, fut):
        timed_log(f'PS start update model')
        delta_tilde = [self.quantize(delta_hat /
                                     self.num_workers) for delta_hat in self.delta_hat]
        for i, e in enumerate(self.error):
            e = self.delta_hat[i] + e - delta_tilde[i]
        for i, p in enumerate(self.model.parameters()):
            p.add_(-delta_tilde[i])
        fut.set_result(delta_tilde)

    @ staticmethod
    @ rpc.functions.async_execution
    def update_model(ps_rref, worker, delta_hat):
        self = ps_rref.local_value()
        timed_log(
            f"PS got {self.curr_update_size+1}/{self.num_workers} updates")
        fut = self.future_model
        with self.lock:
            timed_log(f'PS got update from trainer{worker+1}')
            self.delta_hat += delta_hat
            self.curr_update_size += 1
            if self.curr_update_size >= self.batch_update_size:
                self.update_logic(fut)
                self.curr_update_size = 0
                self.future_model = torch.futures.Future()
        return fut
