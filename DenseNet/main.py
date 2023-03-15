from dataloader import CustomDataset
import torch
import torchvision.transforms as T

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

transforms = torch.nn.Sequential(
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(norm_mean, norm_std)
)

# hyper-parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 5
LR = 0.001
EPOCH = 5

trainset = CustomDataset(image_file='./data/train/train_imgs',
                         label_file='./data/train/train_labels')

testset = CustomDataset(image_file='./data/test/test_imgs',
                        label_file='./data/test/test_labels')

train_iter = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_iter = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


