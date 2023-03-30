import os
import argparse

import torch.multiprocessing as mp

from EfficientAdam_PS import *
from EfficientAdam_Trainer import *
from utils import *

parser = argparse.ArgumentParser(
    description="ResNet 18 Training using CADA on CIFAR10")
parser.add_argument("--resume", "-r", action="store_true",
                    help="Resume from checkpoint")
args = parser.parse_args()


def run_trainer(ps_rref, worker):
    trainer = EfficientAdamTrainer(
        ps_rref=ps_rref, worker=worker, device=devices[worker])
    trainer.train()


def run_ps(trainers):
    timed_log("Start training")
    checkpoint_file = 'EfficientAdam.pt'
    ps_rref = rpc.RRef(EfficientAdamParameterServer(
        device=devices[0], resume_file=(checkpoint_file if args.resume else '')))
    futs = []
    stats_running_file = 'EfficientAdam.csv'
    if not args.resume:
        write_stats_header(stats_running_file, headers=headers)
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
        current_iter = {**eval, **stats}
        write_stats_iter(stats_running_file, current_iter)
        timed_log(f'Finished epoch {e+1}, loss: {loss}, acc: {acc}')
        timed_log(
            f'Current epoch communication rounds: {comms}, bits tranferred: {bits}')

    final_model = ps_rref.rpc_sync().get_model()
    torch.save(final_model.state_dict(), checkpoint_file)
    timed_log("Finish training")


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = MASTER_HOST
    os.environ['MASTER_PORT'] = MASTER_PORT
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
