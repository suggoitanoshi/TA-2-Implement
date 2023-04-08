import argparse

import torch.multiprocessing as mp

from CADA_PS import *
from CADA_Trainer import *
from runner import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ResNet 18 Training using CADA on CIFAR10")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    main(PS=CADAParameterServer, PS_args={}, Trainer=CADATrainer, Trainer_args={
    }, checkpoint_file='CADA.pt', stats_running_file='CADA.csv', args=args)
