import argparse

import torch.multiprocessing as mp

from EfficientAdam_PS import *
from EfficientAdam_Trainer import *
from runner import main

def EfficientAdam_main(args={}):
    learning_rate = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    main(PS=EfficientAdamParameterServer, PS_args={'learning_rate': learning_rate, 'beta_1': beta_1, 'beta_2': beta_2}, Trainer=EfficientAdamTrainer, Trainer_args={'learning_rate': learning_rate, 'beta_1': beta_1, 'beta_2': beta_2}, checkpoint_file='EfficientAdam.pt', stats_running_file='EfficientAdam.csv', args=args)

if __name__ == "__main__":
    EfficientAdam_main()
