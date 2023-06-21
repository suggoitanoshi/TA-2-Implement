from CADA_PS import *
from CADA_Trainer import *
from runner import main

def CADA_main(args={}):
    lr = 0.00001
    beta_1 = 0.9
    beta_2 = 0.99
    delay_bound = 50
    dmax = 10
    c = 400
    main(PS=CADAParameterServer, PS_args={'learning_rate': lr, 'beta_1': beta_1, 'beta_2': beta_2, 'dmax': dmax, 'c': c}, Trainer=CADATrainer, Trainer_args={'delay_bound': delay_bound}, checkpoint_file='CADA.pt', stats_running_file='CADA.csv', args=args)

if __name__ == "__main__":
    CADA_main()
