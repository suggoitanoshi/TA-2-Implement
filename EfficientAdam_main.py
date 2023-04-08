import argparse

import torch.multiprocessing as mp

from EfficientAdam_PS import *
from EfficientAdam_Trainer import *
from runner import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ResNet 18 Training using CADA on CIFAR10")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    main(PS=EfficientAdamParameterServer, PS_args={}, Trainer=EfficientAdamTrainer, Trainer_args={
    }, checkpoint_file='EfficientAdam.pt', stats_running_file='EfficientAdam.csv', args=args)
