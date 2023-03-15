from dataloader import CustomDataset
import torch
import torchvision.transforms as T
from torch import nn
import torch.optim as optim
from densenet import densenet
from train import train, test

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

transforms = torch.nn.Sequential(
    # T.Resize(256),
    # T.CenterCrop(224),
    T.Normalize(norm_mean, norm_std)
)

# hyper-parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 5
LR = 0.001
EPOCH = 5
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
    train(model, train_iter, criterion, optimizer,
          EPOCH, device, num_print, test_iter)
