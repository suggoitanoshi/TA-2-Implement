import argparse

import torch.multiprocessing as mp

from TA_PS import *
from TA_Trainer import *
from utils import *
from runner import main

def TA_main(args={}):
    lr = 0.00001
    beta_1 = 0.9
    beta_2 = 0.99
    delay_bound = 50
    dmax = 10
    c = 400
    main(PS=TAParameterServer, PS_args={'learning_rate': lr, 'beta_1': beta_1, 'beta_2': beta_2, 'dmax': dmax, 'c': c}, Trainer=TATrainer, Trainer_args={'learning_rate': lr, 'beta_1': beta_1, 'beta_2': beta_2, 'delay_bound': delay_bound}, checkpoint_file='TA.pt', stats_running_file='TA.csv', args=args)

if __name__ == "__main__":
    TA_main()
