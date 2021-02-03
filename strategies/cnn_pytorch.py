#import tensorflow as tf
import torch as T
from torch.utils.data import TensorDataset, DataLoader

#from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from tensorflow.keras import Model

import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample

if T.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = T.device(dev)

def process(data):
    df = pd.DataFrame(data, columns=['open','close','low','high'])
    n = 30
    for i in range(1,n+1):
        df['open_'+str(i)] = df['open'].shift(-i)
        df['close_'+str(i)] = df['close'].shift(-i)
        df['low_'+str(i)] = df['low'].shift(-i)
        df['high_'+str(i)] = df['high'].shift(-i)
    period = 30 + n
    df['y'] = df['close'].shift(periods=-period)
    df = df[df['close_'+str(n)]!=df['y']]
    df['y'] = (df['close_'+str(n)] < df['y']).astype(np.int32) # call = 1
    df = df[:-n]
    df = df.to_numpy()
    x,y = df[:,:-1], df[:,-1]
    mx,mn = np.max(x), np.min(x)
    diff = mx-mn
    x -= mn
    x = x/diff
    return x,y

def upsample(x,y):
    df = np.hstack((x,y.reshape(-1,1)))
    # Upsample minority class
    put = df[df[:,-1] == 0]
    call = df[df[:,-1] == 1]
    if len(put) < len(call):
        put = resample(put, replace=True, n_samples=len(call), random_state=1)
    else:
        call = resample(call, replace=True, n_samples=len(put), random_state=1)
    df = np.vstack([put, call])
    # print('0 =',df[df[:,-1]==0][:,-1].shape[0])
    # print('1 =',df[df[:,-1]==1][:,-1].shape[0])
    np.random.shuffle(df)
    return df[:,:-1], df[:,-1]


# train on one data and test on all other data
def one():
    filenames = [i for i in os.listdir() if 'npy' in i and not 'size' in i]
    data = np.load(filenames[0])
    x,y = process(data)
    #x = x*0.1
    ratio = int(y.shape[0] * 0.75)

    x_train,y_train = x,y
    # x_train,x_test = x[:ratio],x[ratio:]
    # y_train,y_test = y[:ratio],y[ratio:]

    x_train = T.Tensor(x_train) # transform to torch tenso
    # r
    # x_test = T.Tensor(x_test) # transform to torch tensor
    y_train = T.Tensor(y_train).to(T.long)
    # y_test = T.Tensor(y_test).to(T.long)

    x_train = x_train.view(-1,11,4).permute(0,2,1)

    #for idx in range(x_train.size(2)):
    #    x_train[:, :, idx] = x_train[:, :, -1] - x_train[:, :, idx]
    # x_test = x_test.view(-1,11,4).permute(0,2,1)

    # print(x_train.size())
    # print(x_test.size())

    train_dataset = TensorDataset(x_train,y_train) # create your datset
    trainloader = DataLoader(train_dataset,
                        batch_size=2,
                        shuffle=True) # create your dataloader
    # test_dataset = TensorDataset(x_test,y_test) # create your datset
    # testloader = DataLoader(test_dataset,
    #                     batch_size=1,
    #                     shuffle=False)

    x_test, y_test = None, None
    for i in range(1,len(filenames)):
        data = np.load(filenames[i])
        x, y = process(data)
        x = T.Tensor(x)
        x = x.view(-1, 11, 4).permute(0, 2, 1)
        print(x.shape)
        #for idx in range(x.size(2)):
        #    x[:,:,idx] = x[:,:,-1] - x[:,:,idx]
        y = T.Tensor(y).to(T.long)
        if i == 1:
            x_test, y_test = x, y
        else:
            x_test, y_test = T.cat([x_test,x]), T.cat([y_test,y])
    test_dataset = TensorDataset(x_test, y_test)
    testloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False)

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print(outputs)
            #print(outputs.size())
            #sys.exit()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            step = 367
            if i % step == step-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / step))
                running_loss = 0.0
    print('[TRAIN]')
    correct = 0
    total = 0
    pred = []
    gt = []
    earn = 0
    loose = 0
    with T.no_grad():
        # for data in testloader:
        for data in trainloader:
            inputs, labels = data
            outputs = net(inputs.to(device))
            _, predicted = T.max(outputs.data, 1)
            pred += predicted.tolist()
            gt += labels.tolist()
            for p,l in zip(predicted,labels):
                if p == l:
                    earn += 0.7
                else:
                    loose += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('pred 1:', sum(pred), ' 0:', len(pred) - sum(pred))
    print('  gt 1:', sum(gt), ' 0:', len(gt) - sum(gt))
    print('      earn:', round(earn))
    print('     loose:', loose)
    print('net profit:', round(earn - loose))

    print('Accuracy of the network on the %d test images: %d %%' % (y_test.size(0),
                                                                    100 * correct / total))
    print()
    print('[TEST]')
    correct = 0
    total = 0
    pred = []
    gt = []
    earn = 0
    loose = 0
    with T.no_grad():
        #for data in testloader:
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs.to(device))
            _, predicted = T.max(outputs.data, 1)
            pred += predicted.tolist()
            gt += labels.tolist()
            if predicted == labels:
                earn += 0.7
            else:
                loose += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('pred 1:', sum(pred), ' 0:',len(pred)-sum(pred))
    print('  gt 1:', sum(gt), ' 0:',len(gt)-sum(gt))
    print('      earn:', round(earn))
    print('     loose:', loose)
    print('net profit:', round(earn-loose))

    print('Accuracy of the network on the %d test images: %d %%' % (y_test.size(0),
        100 * correct / total))

    print('Finished Training')

def two():
    total_earn = 0
    total_loose = 0
    filenames = [i for i in os.listdir() if 'npy' in i and not 'size' in i]
    for i in range(1,len(filenames)):
        data = np.load(filenames[i])
        x,y = process(data)
        ratio = int(y.shape[0] * 0.75)
        x_train,x_test = x[:ratio],x[ratio:]
        y_train,y_test = y[:ratio],y[ratio:]

        x_train,y_train = upsample(x_train,y_train)

        x_train = T.Tensor(x_train) # transform to torch tenso
        x_test = T.Tensor(x_test) # transform to torch tensor
        y_train = T.Tensor(y_train).to(T.long)
        y_test = T.Tensor(y_test).to(T.long)

        x_train = x_train.view(-1,31,4).permute(0,2,1)
        x_test = x_test.view(-1,31,4).permute(0,2,1)

        train_dataset = TensorDataset(x_train,y_train) # create your datset
        trainloader = DataLoader(train_dataset,
                            batch_size=2,
                            shuffle=True) # create your dataloader
        test_dataset = TensorDataset(x_test,y_test) # create your datset
        testloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False)

        net = Net()
        net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in tqdm(range(10)):
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                # running_loss += loss.item()
                # step = len(trainloader)
                # if i % step == step-1:    # print every 2000 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, running_loss / step))
                #     running_loss = 0.0
        print('[TRAIN]')
        correct = 0
        total = 0
        pred = []
        gt = []
        earn = 0
        loose = 0
        with T.no_grad():
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs.to(device))
                _, predicted = T.max(outputs.data, 1)
                pred += predicted.tolist()
                gt += labels.tolist()
                for p,l in zip(predicted,labels):
                    if p == l:
                        earn += 0.7
                    else:
                        loose += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('pred 1:', sum(pred), ' 0:', len(pred) - sum(pred))
        print('  gt 1:', sum(gt), ' 0:', len(gt) - sum(gt))
        print('      earn:', round(earn))
        print('     loose:', loose)
        print('net profit:', round(earn - loose))

        print('Accuracy of the network on the %d test images: %d %%' % (y_test.size(0),
                                                                        100 * correct / total))
        print('[TEST]')
        correct = 0
        total = 0
        pred = []
        gt = []
        earn = 0
        loose = 0
        with T.no_grad():
            #for data in testloader:
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = T.max(outputs.data, 1)
                pred += predicted.tolist()
                gt += labels.tolist()
                if predicted == labels:
                    earn += 0.7
                else:
                    loose += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        total_earn += earn
        total_loose += loose
        print('pred 1:', sum(pred), ' 0:',len(pred)-sum(pred))
        print('  gt 1:', sum(gt), ' 0:',len(gt)-sum(gt))
        print('      earn:', round(earn))
        print('     loose:', loose)
        print('net profit:', round(earn-loose))
        print('Accuracy of the network on the %d test images: %d %%\n' % (y_test.size(0),
            100 * correct / total))
        print('='*40)
        # break
    print('Finished Training\n\n')
    print('total earn:', round(total_earn))
    print('total loose:', total_loose)
    print('total net profit:', round(total_earn-total_loose))

def three():
    x_train, y_train, x_test, y_test = None, None, None, None
    filenames = [i for i in os.listdir() if 'npy' in i and not 'size' in i]
    for i in range(1,len(filenames)):
        data = np.load(filenames[i])
        x,y = process(data)
        ratio = int(y.shape[0] * 0.75)
        x_train_,x_test_ = x[:ratio],x[ratio:]
        y_train_,y_test_ = y[:ratio],y[ratio:]

        if i == 1:
            x_train = x_train_
            y_train = y_train_
            x_test = x_test_
            y_test = y_test_
        else:
            x_train = np.vstack([x_train,x_train_])
            y_train = np.hstack([y_train, y_train_])
            x_test = np.vstack([x_test,x_test_])
            y_test = np.hstack([y_test,y_test_])
        break

    print(x_train.shape, x_train_.shape)
    print(y_train.shape, y_train_.shape)
    print(x_test.shape, x_test_.shape)
    print(y_test.shape, y_test_.shape)
    x_train,y_train = upsample(x_train,y_train)
    print('='*40)
    print(x_train.shape, x_train_.shape)
    print(y_train.shape, y_train_.shape)
    print(x_test.shape, x_test_.shape)
    print(y_test.shape, y_test_.shape)

    x_train = T.Tensor(x_train) # transform to torch tenso
    x_test = T.Tensor(x_test) # transform to torch tensor
    y_train = T.Tensor(y_train).to(T.long)
    y_test = T.Tensor(y_test).to(T.long)

    x_train = x_train.view(-1,31,4).permute(0,2,1)
    x_test = x_test.view(-1,31,4).permute(0,2,1)
    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    trainloader = DataLoader(train_dataset,
                             batch_size=2,
                             shuffle=True)  # create your dataloader
    test_dataset = TensorDataset(x_test, y_test)  # create your datset
    testloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False)

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(1)):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print('[TRAIN]')
    correct = 0
    total = 0
    pred = []
    gt = []
    earn = 0
    loose = 0
    with T.no_grad():
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs.to(device))
            _, predicted = T.max(outputs.data, 1)
            pred += predicted.tolist()
            gt += labels.tolist()
            for p, l in zip(predicted, labels):
                if p == l:
                    earn += 0.7
                else:
                    loose += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('pred 1:', sum(pred), ' 0:', len(pred) - sum(pred))
    print('  gt 1:', sum(gt), ' 0:', len(gt) - sum(gt))
    print('      earn:', round(earn))
    print('     loose:', loose)
    print('net profit:', round(earn - loose))

    print('Accuracy of the network on the %d test images: %d %%' % (y_test.size(0),
                                                                    100 * correct / total))
    print('[TEST]')
    correct = 0
    total = 0
    pred = []
    gt = []
    earn = 0
    loose = 0
    sm = nn.Softmax(0)
    with T.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = T.max(outputs.data, 1)
            pred += predicted.tolist()
            gt += labels.tolist()
            confidence,_ = T.max(sm(outputs.data),1)
            if confidence < 0.54:
                continue
            print(sm(outputs),confidence,predicted==labels)
            if predicted == labels:
                earn += 0.7
            else:
                loose += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('pred 1:', sum(pred), ' 0:', len(pred) - sum(pred))
    print('  gt 1:', sum(gt), ' 0:', len(gt) - sum(gt))
    print('      earn:', round(earn))
    print('     loose:', loose)
    print('net profit:', round(earn - loose))
    print('Accuracy of the network on the %d test images: %d %%\n' % (y_test.size(0),
                                                                      100 * correct / total))
    print('=' * 40)

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(4, 4**5, 3, 1, 1)
        self.conv2 = nn.Conv1d(4**5, 4**4, 3, 1, 1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(1792,64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16,2)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = T.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    def p(self, x):
        print(x,x.size())
        print()


if __name__ == '__main__':
    # one()
    # two()
    three()