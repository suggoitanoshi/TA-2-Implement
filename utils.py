from datetime import datetime
import torch
from torchvision import transforms
import logging
import sys
import csv

batch_size = 100
image_w = 64
image_h = 64
num_classes = 10
batch_update_size = 5
nsample = 5000
delay_bound = 50
epochs = 50
learning_rate = .01
lr_decay = 0.5
beta_1 = 0.9
beta_2 = 0.99
c = 0.12*5
dmax = 2
epsilon = 1e-6

device_count = torch.cuda.device_count()
devices = [torch.device('cuda', i % device_count) if torch.cuda.is_available(
) else torch.device('cpu') for i in range(batch_update_size)]

headers = ['loss', 'acc', 'comms', 'bits']

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])


def timed_log(text):
    logger.info(f"{datetime.now().strftime('%H:%M:%S')} {text}")


def quantize(v, device='cpu', num_bits=16):
    v_norm = torch.norm(v)
    if v_norm < 1e-10:
        qv = torch.zeros_like(v).to(device)
    else:
        s = 2**(num_bits-1)
        l = torch.floor(torch.abs(v)/v_norm*s)
        p = torch.abs(v)/v_norm-l
        qv = v_norm*torch.sign(v)*(l/s + l/s*(torch.rand_like(v) < p).float())
    return qv


def write_stats_header(outfile, headers):
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()


def write_stats_iter(outfile, iter_data):
    with open(outfile, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=iter_data.keys())
        writer.writerow(iter_data)


def write_stats(outfile, all_epoch_data):
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_epoch_data[0].keys())
        writer.writeheader()
        writer.writerows(all_epoch_data)


def collate_train(data):
    imgs, labels = zip(*data)
    imgs = torch.stack([transform_train(img) for img in imgs])
    return imgs, torch.tensor(labels)
