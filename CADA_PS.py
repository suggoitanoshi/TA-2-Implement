from utils import *
from parameter_server import BatchUpdateParameterServer


class CADAParameterServer(BatchUpdateParameterServer):
    def __init__(self, device, batch_update_size=batch_update_size, num_workers=batch_update_size, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, c=c, dmax=dmax):
        super().__init__(device=device, batch_update_size=batch_update_size, num_workers=num_workers,
                         learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        self.triggerlist = [0 for _ in range(dmax)]
        self.thrd_scale = c/dmax

    def update_logic(self, fut):
        timed_log(f'PS start update model')
        diff = super().update_logic(fut)
        self.triggerlist.append(diff)
        self.triggerlist.pop(0)
        thrd = sum(self.triggerlist)*self.thrd_scale
        fut.set_result(thrd)
