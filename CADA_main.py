import os

import torch.multiprocessing as mp

from CADA_PS import *
from CADA_Trainer import *
from utils import *


def run_trainer(ps_rref, worker):
    trainer = CADATrainer(ps_rref=ps_rref, worker=worker,
                          device=devices[worker])
    trainer.train()


def run_ps(trainers):
    timed_log("Start training")
    ps_rref = rpc.RRef(CADAParameterServer(device=devices[0]))
    futs = []
    all_epoch_data = []
    for e in range(epochs):
        timed_log(f'Start epoch {e+1}/{epochs}')
        for i, trainer in enumerate(trainers):
            futs.append(
                rpc.rpc_async(trainer, run_trainer, args=(ps_rref, i))
            )
        torch.futures.wait_all(futs)
        futs = []
        eval = ps_rref.rpc_sync().eval()
        stats = ps_rref.rpc_sync().get_stats()
        ps_rref.rpc_sync().reset_stats()
        loss = eval['loss']
        acc = eval['acc']
        comms = stats['comms']
        bits = stats['bits']
        all_epoch_data.append({**eval, **stats})
        timed_log(f'Finished epoch {e+1}, loss: {loss}, acc: {acc}')
        timed_log(
            f'Current epoch communication rounds: {comms}, bits tranferred: {bits}')

    write_stats('CADA.csv', all_epoch_data=all_epoch_data)

    timed_log("Finish training")


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0  # infinite timeout
    )
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)])

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = batch_update_size + 1
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)
