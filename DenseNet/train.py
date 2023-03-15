

def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None, test_iter=None):
    net.train()
    record_train = list()
    record_test = list()
    

    for epoch in range(num_epochs):
        print(
            "========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))
        total, correct, train_loss = 0, 0, 0
        start = time.time()

        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            output = net(X)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            train_acc = 100.0 * correct / total

            if (i + 1) % num_print == 0:
                print("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}"
                      .format(i + 1, len(train_iter), train_loss / (i + 1),
                              train_acc, get_cur_lr(optimizer)))

        if lr_scheduler is not None:
            lr_scheduler.step()

        print("--- cost time: {:.4f}s ---".format(time.time() - start))

        if test_iter is not None:
            record_test.append(test(net, test_iter, criterion, device))
        record_train.append(train_acc)

    return record_train, record_test