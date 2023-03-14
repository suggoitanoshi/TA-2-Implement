from datetime import datetime
import torch

batch_size = 200
image_w = 64
image_h = 64
num_classes = 30
batch_update_size = 5
num_batches = 1
delay_bound = 50
epochs = 5
learning_rate = .01
beta_1 = 0.9
beta_2 = 0.99
c = 0.12*5
dmax = 2
device_count = torch.cuda.device_count()
devices = [torch.device('cuda', i % device_count) if torch.cuda.is_available() else torch.device('cpu') for i in range(batch_update_size)]


def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


def sort_idx(dataset, num_classes):
    sorted = [[] for i in range(num_classes)]
    for i in range(len(dataset)):
        sorted[dataset[i][1]].append(i)
    sorted_idx = []
    for i in range(num_classes):
        sorted_idx = sorted_idx + sorted[i]
    return sorted_idx


def quantize(v, num_bits=16):
    v_norm = torch.norm(v)
    if v_norm < 1e-10:
        qv = 0
    else:
        s = 2**(num_bits-1)
        l = torch.floor(torch.abs(v)/v_norm*s)
        p = torch.abs(v)/v_norm-l
        qv = v_norm*torch.sign(v)*(l/s + l/s*(torch.rand_like(v) < p).float())
    return qv
