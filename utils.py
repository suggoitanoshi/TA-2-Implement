from datetime import datetime
import torch
from torchvision import transforms
import logging
import sys
import csv

batch_size = 50
image_w = 64
image_h = 64
num_classes = 10
batch_update_size = 10
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

headers = ['loss', 'acc', 'comms', 'bits', 'time']

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(24, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                      (0.2470, 0.2435, 0.2616)),
    transforms.Normalize((0.1307), (0.3081))
])

transform_test = transforms.Compose([
    transforms.RandomCrop(24, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                      (0.2470, 0.2435, 0.2616)),
    transforms.Normalize((0.1307), (0.3081))
])


def timed_log(text):
    logger.info(f"{datetime.now().strftime('%H:%M:%S')} {text}")


def __construct_quant_M(k, K):
    return torch.pow(torch.ones((K-k+1,))*2, torch.tensor(list(range(k, K+1))))


M = __construct_quant_M(-17, -11)


@torch.no_grad()
def quantize(v, device='cpu'):
    # __M = M.to(device)
    # x, y = torch.meshgrid(v.reshape(-1), __M, indexing='ij')
    # idx = torch.argmin(torch.abs(x - y), 1)
    # return __M[idx].reshape(v.shape).clone().to(device)
    return v.clone().to(device=device, dtype=torch.float16)


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


def sort_idx(dataset, num_classes, num_samples):
    sorted = [[] for i in range(num_classes)]
    for i in range(num_samples):
        sorted[dataset[i][1]].append(i)
    sorted_idx = []
    for i in range(num_classes):
        sorted_idx = sorted_idx + sorted[i]
    return sorted_idx
