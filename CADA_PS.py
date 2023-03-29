from utils import *
from parameter_server import BatchUpdateParameterServer


class CADAParameterServer(BatchUpdateParameterServer):
    def __init__(self, device, batch_update_size=batch_update_size, num_workers=batch_update_size, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, c=c, dmax=dmax, resume_file=''):
        super().__init__(device=device, batch_update_size=batch_update_size, num_workers=num_workers,
                         learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, resume_file=resume_file)
        self.grad = [None for _ in range(num_workers)]
        self.triggerlist = [0 for _ in range(dmax)]
        self.thrd_scale = c/dmax

    def _update_model(self, worker, data):
        fut = self.future_model
        with self.lock:
            if data['grad'] is not None:
                timed_log(f'PS got update from trainer{worker+1}')
                self.grad[worker] = data['grad']
                self.add_comm_curr_epoch()
                self.add_bits_curr_epoch(
                    sum([grad.nelement() * grad.element_size() for grad in data['grad'] if grad is not None]))
            self.curr_update_size += 1
            if self.curr_update_size >= self.batch_update_size:
                self.update_logic(fut)
                self.curr_update_size = 0
                self.future_model = torch.futures.Future()
        return fut

    def update_logic(self, fut):
        timed_log(f'PS start update model')
        diff = super().update_logic(fut)
        self.triggerlist.append(diff)
        self.triggerlist.pop(0)
        thrd = sum(self.triggerlist)*self.thrd_scale
        self.add_bits_curr_epoch(64)
        fut.set_result(thrd)
