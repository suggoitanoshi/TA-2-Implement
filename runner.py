import os

import torch.multiprocessing as mp
from torch.distributed import rpc
import portpicker

from utils import *


def run_trainer(Trainer, ps_rref, worker, Trainer_args):
    trainer = Trainer(ps_rref=ps_rref, worker=worker,
                      device=devices[worker], **Trainer_args)
    trainer.train()


def run_ps(trainers, PS, PS_args, Trainer, Trainer_args, stats_running_file, checkpoint_file, args):
    timed_log("Start training")
    ps_rref = rpc.RRef(PS(
        device=devices[0], resume_file=(checkpoint_file if args.resume else ''), **PS_args))
    futs = []
    if not args.resume:
        write_stats_header(stats_running_file, headers=headers)

    lr = learning_rate
    for e in range(epochs):
        if e > 0 and e % 50 == 0:
            lr *= lr_decay
            ps_rref.rpc_sync().set_learning_rate(lr)
        timed_log(f'Start epoch {e+1}/{epochs}')
        for i, trainer in enumerate(trainers):
            futs.append(
                rpc.rpc_async(trainer, run_trainer, args=(
                    Trainer, ps_rref, i, Trainer_args))
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
        if e > 0 and e % 50 == 0:
            torch.save(ps_rref.rpc_sync().serialize(), checkpoint_file)

    timed_log("Finish training")
    torch.save(ps_rref.rpc_sync().serialize(), checkpoint_file)


def run(rank, world_size, PS, PS_args, Trainer, Trainer_args, stats_running_file, checkpoint_file, args):
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=4,
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
        run_ps([f"trainer{r}" for r in range(1, world_size)], PS=PS,
               PS_args=PS_args, Trainer=Trainer, Trainer_args=Trainer_args, stats_running_file=stats_running_file, checkpoint_file=checkpoint_file, args=args)

    # block until all rpcs finish
    rpc.shutdown()


def main(PS, PS_args, Trainer, Trainer_args, stats_running_file, checkpoint_file, args):
    MASTER_ADDR = os.environ.get('MASTER_ADDR', 'localhost')
    MASTER_PORT = os.environ.get(
        'MASTER_PORT', str(portpicker.pick_unused_port()))
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    world_size = batch_update_size + 1
    mp.spawn(run, args=(world_size, PS, PS_args, Trainer,
             Trainer_args, os.path.join('results', stats_running_file), os.path.join('results', checkpoint_file), args), nprocs=world_size, join=True)
