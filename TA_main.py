import argparse

import torch.multiprocessing as mp

from TA_PS import *
from TA_Trainer import *
from utils import *
from runner import main

def TA_main(args={}):
    main(PS=TAParameterServer, PS_args={}, Trainer=TATrainer, Trainer_args={}, checkpoint_file='TA.pt', stats_running_file='TA.csv', args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ResNet 18 Training using TA on CIFAR10")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()
    TA_main(args)
