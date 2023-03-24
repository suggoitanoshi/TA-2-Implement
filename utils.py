from datetime import datetime
import torch
import logging
import sys
import csv

batch_size = 1000
image_w = 64
image_h = 64
num_classes = 10
batch_update_size = 5
nsample = 5000
delay_bound = 50
epochs = 50
learning_rate = .01
beta_1 = 0.9
beta_2 = 0.99
c = 0.12*5
dmax = 2
epsilon = 1e-6
device_count = torch.cuda.device_count()
devices = [torch.device('cuda', i % device_count) if torch.cuda.is_available(
) else torch.device('cpu') for i in range(batch_update_size)]

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


def timed_log(text):
    logger.info(f"{datetime.now().strftime('%H:%M:%S')} {text}")


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
        qv = torch.zeros_like(v)
    else:
        s = 2**(num_bits-1)
        l = torch.floor(torch.abs(v)/v_norm*s)
        p = torch.abs(v)/v_norm-l
        qv = v_norm*torch.sign(v)*(l/s + l/s*(torch.rand_like(v) < p).float())
    return qv


def write_stats(outfile, all_epoch_data):
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_epoch_data[0].keys())
        writer.writeheader()
        writer.writerows(all_epoch_data)
