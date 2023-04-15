#    Horovod
#    Copyright 2018 Uber Technologies, Inc.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import print_function
import argparse
from dataset import CustomDataset
from densenet import densenet
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd
import numpy as np
import time

criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    count_time = list()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (img, label) in enumerate(train_loader):
        if args.cuda:
            img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(img.to(torch.float32))
        loss = criterion(output, label)
        loss.backward()
        starttime = time.time()
        optimizer.step()  ## this obj is produced by hvd, so gradient allreduced inside
        endtime = time.time()
        count_time.append(endtime - starttime)
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx,
                    len(train_sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    count_time = np.sum(count_time) / len(count_time)
    print("the average epoch time of this {} is {:.2f}".format(epoch, count_time))
    return count_time


def test():
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    for img, label in test_loader:
        if args.cuda:
            img, label = img.cuda(), label.cuda()
        output = model(img.to(torch.float32))

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(label.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    # print(hvd.rank(), 'test_loss=', test_loss, 'test_accuracy=', test_accuracy))
    test_accuracy = metric_average(test_accuracy, "avg_accuracy")

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print("\nTest Accuracy: {:.2f}%\n".format(100.0 * test_accuracy))


# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for testing (default: 60)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce",
)
parser.add_argument(
    "--use-adasum",
    action="store_true",
    default=False,
    help="use adasum algorithm to do reduction",
)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

transforms = torch.nn.Sequential(
    T.Resize([224, 224]),
    # T.CenterCrop(224),
    T.Normalize(norm_mean, norm_std),
)
# hyper-parameters
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (
        kwargs.get("num_workers", 0) > 0
        and hasattr(mp, "_supports_context")
        and mp._supports_context
        and "forkserver" in mp.get_all_start_methods()
    ):
        kwargs["multiprocessing_context"] = "forkserver"

    trainset = CustomDataset(
        image_file="data/train_imgs.txt",
        label_file="data/train_labels.txt",
        transform=transforms,
    )

    testset = CustomDataset(
        image_file="data/test_imgs.txt",
        label_file="data/test_labels.txt",
        transform=transforms,
    )

    # Horovod: use DistributedSampler to partition the training data.
    ## this is a professional class of parti data intead of coding manually
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
    )

    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs
    )

    model = densenet(in_channel=3, num_classes=5)

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr * lr_scaler, momentum=args.momentum
    )  ## standard optimizer

    # Horovod: broadcast parameters & optimizer state.
    ## hvd use bc since 1st time, from root to all workers??
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    ## hvd use grad compress to save comm bandwidth
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    ## turn standard optim into hvd's optim, so grad allreduce inside
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
    )

    # device = 'cuda' if args.cuda else 'cpu'
    starttime = time.time()
    average_count_time = 0
    for epoch in range(1, args.epochs + 1):
        average_count_time += train(epoch)
        # test()
    average_count_time /= args.epochs
    endtime = time.time()
    print("average step time {:.2d}".format(average_count_time))
    print("average epoch time {:.2f}".format((endtime - starttime) / args.epochs))
