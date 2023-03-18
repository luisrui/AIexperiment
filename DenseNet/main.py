from dataloader import CustomDataset
import torch
import torchvision.transforms as T
from torch import nn
import torch.optim as optim
from densenet import densenet
from train import train, test
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

transforms = torch.nn.Sequential(
    # T.Resize(256),
    # T.CenterCrop(224),
    T.Normalize(norm_mean, norm_std)
)

# hyper-parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 10
LR = 0.001
EPOCH = 10
num_print = 10
criterion = nn.CrossEntropyLoss()

model = densenet(in_channel=3, num_classes=50)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

trainset = CustomDataset(image_file='D:\\AIexperiment\\DenseNet\data\\train\\train_imgs.txt',
                         label_file='D:\\AIexperiment\\DenseNet\data\\train\\train_labels.txt')

testset = CustomDataset(image_file='D:\\AIexperiment\\DenseNet\\data\\test\\test_imgs.txt',
                        label_file='D:\\AIexperiment\\DenseNet\\data\\test\\test_labels.txt')

train_iter = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_iter = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

if __name__ == '__main__':
    record_loss, record_train, record_test = train(model, train_iter, criterion, optimizer,
                                                   EPOCH, device, num_print, test_iter)
    torch.save(model, 'D:\\AIexperiment\\DenseNet\\savedmodels\\first_10.pth')
    input = torch.ones((64, 3, 32, 32))
    output = model(input)

    writer = SummaryWriter('D:\\AIexperiment\\DenseNet\\result')
    writer.add_graph(model, input)
    for i in range(len(record_loss)):
        writer.add_scalar("Train loss", record_loss[i], i)
    for i in range(len(record_train)):
        writer.add_scalar("Train accuracy", record_train[i], i)
    for i in range(len(record_test)):
        writer.add_scalar("Test accuracy", record_test[i], i)
    writer.close()
    # print('record_test : ', record_test)
    # print('record_train : ', record_train)
    # plt.subplot(1, 2, 1)
    # plt.plot(np.arange(len(record_train)) + 1, record_train)
    # plt.title = 'train accuracy'
    # plt.ylabel = 'accuracy'
    # plt.xlabel = 'record times'

    # plt.subplot(1, 2, 2)
    # plt.plot(np.arange(len(record_test)) + 1, record_test)
    # plt.title = 'test accuracy'
    # plt.ylabel = 'accuracy'
    # plt.xlabel = 'epochs'

    # plt.savefig('./result_10_epoch.jpg')
