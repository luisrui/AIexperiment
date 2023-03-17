import time
from torch import nn
import torch


def test(net, test_iter, device):
    total, correct = 0, 0
    for i, (img, label) in enumerate(test_iter):
        img, label = img.to(device), label.to(device)
        output = net(img.to(torch.float32))

        total += label.size(0)
        correct += (output.argmax(dim=1) == label).sum().item()
    test_acc = 100 * correct / total
    return test_acc


def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, test_iter=None):
    net.train()
    record_train = list()
    record_test = list()
    record_loss = list()

    for epoch in range(num_epochs):
        print(
            "========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        total, correct, train_loss = 0, 0, 0
        start = time.time()

        for i, (img, label) in enumerate(train_iter):
            img, label = img.to(device), label.to(device)
            output = net(img.to(torch.float32))
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += label.size(0)
            correct += (output.argmax(dim=1) == label).sum().item()
            train_acc = 100.0 * correct / total

            if (i + 1) % num_print == 0:
                print("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}"
                      .format(i + 1, len(train_iter), train_loss / (i + 1),
                              train_acc, optimizer.state_dict()['param_groups'][0]['lr']))
            record_loss.append(train_loss / (i + 1))
        # if lr_scheduler is not None:
        #     lr_scheduler.step()

        print("--- cost time: {:.4f}s ---".format(time.time() - start))

        if test_iter is not None:
            record_test.append(test(net, test_iter, device))
        record_train.append(train_acc)

    return record_loss, record_train, record_test
