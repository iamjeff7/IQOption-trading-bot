import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from datetime import datetime
import utils
import fire
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)


def process(d):
    nc0 = d[0:-3, :]
    nc1 = d[1:-2, :]
    nc2 = d[2:-1, :]
    nc3 = d[3:, 1]
    y = np.array([1 if i[1] < j else 0 for i,j in zip(nc2, nc3)])
    #for a,b,c,d,e in zip(nc0,nc1,nc2,nc3,y):
        #print(f'{a:.5f} {b:.5f} {c:.5f} {d:.5f} : {e}')
        #print(f'{a} {b} {c} {d} : {e}')
        #break
    x = np.hstack([nc0,nc1,nc2])
    return x, y

def processx(d):
    x = d.reshape(1,-1)
    return x

def MinMaxScaler(d):
    mn, mx = np.min(d, axis=1, keepdims=True), np.max(d, axis=1, keepdims=True)
    mn, mx = mn-0.00001, mx+0.00001
    diff = mx-mn
    #d = d[:, [0,1,4,5,8,9]]
    d -= mn
    d = d/diff
    new_mn, new_mx = -1, 1
    new_diff = new_mx - new_mn
    d *= new_diff
    d += new_mn
    return d

def downsample(x,y):
    df = np.hstack((x,y.reshape(-1,1)))
    # Downsample minority class
    put = df[df[:,-1] == 0]
    call = df[df[:,-1] == 1]
    if len(put) > len(call):
        put = resample(put, replace=True, n_samples=len(call), random_state=1)
    else:
        call = resample(call, replace=True, n_samples=len(put), random_state=1)
    df = np.vstack([put, call])
    np.random.shuffle(df)
    return df[:,:-1], df[:,-1]

class Net(torch.nn.Module):
    def __init__(self, in_channels, num_class=2, features=100):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, features)
        self.fc = torch.nn.Linear(features, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

class LSTM_Net(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=100, batch_size=2**10, output_dim=2,
                    num_layers=2):
        super(LSTM_Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.3)
        # Define the output layer
        self.linear = torch.nn.Linear(self.hidden_dim, output_dim)
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                        torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = x.view(-1, 3, 4).permute(1, 0, 2)
        x, self.hidden = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = torch.max(x, 1)[0]
        x = self.linear(x)
        return x

class LSTM_Net2(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=100, batch_size=2**10, output_dim=2,
                    num_layers=2):
        super(LSTM_Net2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.3)
        # Define the output layer
        self.linear = torch.nn.Linear(self.hidden_dim*3, output_dim)
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                        torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x, self.hidden = self.lstm(x.view(-1, 3, 4).permute(1, 0, 2))
        x = self.flatten(x.permute(1, 0, 2))
        x = self.linear(x)
        return x


def test():
    import os
    import sys
    sys.path.append(os.getcwd())
    from practice_data import PracticeData2

    ptd = PracticeData2('EURUSD', 1)
    data = ptd.data
    x, y = process(data)
    x = MinMaxScaler(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y,
                                                        shuffle=True)
    x_train, y_train = downsample(x_train, y_train)

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train).to(torch.long)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test).to(torch.long)

    train_dataset = TensorDataset(x_train,y_train)
    trainloader = DataLoader(train_dataset,
                        batch_size=1,
                        shuffle=True)
    test_dataset = TensorDataset(x_test,y_test)
    testloader = DataLoader(test_dataset,
                        batch_size=1,
                        shuffle=False)

    '''
    model = Net(x_train.size(1), 2)
    if str(device) == 'cpu':
        model.load_state_dict(torch.load('strategies/model_weights/model.pth', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('strategies/model_weights/model.pth'))

    '''

    model = LSTM_Net(input_dim=4, hidden_dim=100, batch_size=1,
                    output_dim=2, num_layers=2).to(device) 
    if str(device) == 'cpu':
        #model.load_state_dict(torch.load('strategies/model_weights/model.pth', map_location=torch.device('cpu')))
        model.load_state_dict(torch.load('strategies/model_weights/model_lstm.pth', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('strategies/model_weights/model.pth'))

    print(f'\n{model}\n')

    for dtx, dataloader in zip(['[TRAIN]', '[TEST]'],[trainloader, testloader]):
        print()
        print(dtx)
        correct = 0
        total = 0
        pred = []
        gt = []
        earn = 0
        loose = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
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
        print('  accuracy:', round(correct/total, 4))
        print()

def train(resume):
    import os
    import sys
    sys.path.append(os.getcwd())
    from practice_data import PracticeData2

    ptd = PracticeData2('EURUSD', 1)
    data = ptd.data
    x, y = process(data)
    x = MinMaxScaler(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y,
                                                        shuffle=True)
    x_train, y_train = downsample(x_train, y_train)

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train).to(torch.long)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test).to(torch.long)

    train_dataset = TensorDataset(x_train,y_train)
    trainloader = DataLoader(train_dataset,
                        batch_size=2**10,
                        shuffle=True)
    test_dataset = TensorDataset(x_test,y_test)
    testloader = DataLoader(test_dataset,
                        batch_size=1,
                        shuffle=False)

    model = Net(x_train.size(1), 2).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('\n\n')
    epochs = 10000
    min_loss = 9999
    last_improve_epoch = 0
    patience = 3
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs,labels) in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            step = len(trainloader) // 100
            if i % step == step-1:
                if running_loss / step < min_loss:
                    min_loss = running_loss / step
                    torch.save(model.state_dict(), 'strategies/model_weights/model_'+str(round(min_loss,6))+'.pth')
                    last_improve_epoch = epoch
                else:
                    if epoch > 100 and epoch - last_improve_epoch > 100:
                        if patience == 0:
                            break
                        else:
                            lr = lr * 0.1
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            patience -= 1
                            last_improve_epoch = epoch
                print('[%3d, %5d] loss: %.4f | min_loss: %.4f | last_improved: %3d' %
                      (epoch + 1, i + 1, running_loss / step, min_loss,
                          last_improve_epoch),
                      end='\r')
                running_loss = 0.0
    test()



def back_test():
    import os
    import sys
    sys.path.append(os.getcwd())
    from practice_data import PracticeData2

    ptd = PracticeData2('EURUSD', 1)
    data = ptd.data
    x, y = process(data)
    x = MinMaxScaler(x)
    print()
    #print(type(data))
    #print(data.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y,
                                                        shuffle=True)

    x_train, y_train = downsample(x_train, y_train)

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]
    longest_name = max([len(name) for name in names])

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2),
        MLPClassifier(activation='logistic', 
            hidden_layer_sizes=(32),
            alpha=0.1, max_iter=1000,
            verbose=True,
            early_stopping=False),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    print('Train')
    print(x_train[:3])
    print('Test')
    print(x_test[:3])
    a, b = np.unique(y_train, return_counts=True)
    print('train:', a, b)
    a, b = np.unique(y_test, return_counts=True)
    print(' test:', a, b)
    # iterate over classifiers
    model_idx = 0
    temp_score = -999

    '''
    for idx, (name, clf) in enumerate(zip(names, classifiers)):
        if name in ['RBF SVM', 'Linear SVM']:
            continue
        print(name, end='\r')
        st = time.time()
        clf.fit(x_train, y_train)
        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        pred = clf.predict(x_test)
        a, b = np.unique(pred, return_counts=True)
        et = time.time()
        #score = 0.1234
        if test_score > temp_score:
            temp_score = test_score
            model_idx = idx
        print(f' {name:<{longest_name}} : {train_score:.4f} : {test_score:.4f} : {a} {b}')
    '''

    clf = MLPClassifier(activation='tanh', 
                hidden_layer_sizes=(2**8),
                learning_rate_init=0.00001,
                alpha=0.0001, max_iter=2000,
                verbose=True,
                n_iter_no_change=100,
                early_stopping=False,
                random_state=2)

    print(f'train data:\n{x_train[0]}\n')
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    pred = clf.predict(x_test)
    a, b = np.unique(pred, return_counts=True)
    print(f' Neural Net | {train_score:.4f} | {test_score:.4f} | {a} | {b}')
    a, b = np.unique(y_train, return_counts=True)
    print('train:', a, b)
    a, b = np.unique(y_test, return_counts=True)
    print(' test:', a, b)
    return clf

    #model = classifiers[model_idx]
    #score = model.score(x_test, y_test)
    #print(f'{model_idx} {names[model_idx]}: {score:.4f}')
    #return classifiers[model_idx]

def final(iq, init_balance):
    term_balance = iq.get_balance()
    max_len = max(len(str(init_balance)), len(str(term_balance)))
    net_profit = term_balance - init_balance
    percent = (net_profit / init_balance) * 100
    print('-'*40)
    print(f'  initial_balance: {init_balance:>{max_len}.2f}\n')
    print(f' terminal_balance: {term_balance:>{max_len}.2f}\n')
    print(f'       net_profit: {net_profit:>{max_len}.2f} USD\n')
    print(f'       net_profit: {net_profit * 30.09:>{max_len}.2f} THB\n')
    print(f'percentage_profit: {percent:>{max_len}.2f} %\n')
    print('-'*40)

def run(init_iter, display_total_earn=True, model=None):
    init_iter = int(init_iter)
    display_total_earn = display_total_earn in ['True']

    print('init_iter:', init_iter, type(init_iter))
    import os
    import sys
    sys.path.append(os.getcwd())
    import utils
    from data import IQOption
    iq = IQOption(goal='EURUSD',
                  size=60,
                  maxdict=3,
                  money=1,
                  expiration_mode=1,
                  account='PRACTICE')
    init_balance = iq.get_balance()
    print('init_balance =', init_balance)

    #init_iter = 60*1
    iter = init_iter

    #test()

    #model = Net(12, 2)
    model = LSTM_Net(input_dim=4, hidden_dim=100, batch_size=1,
                    output_dim=2, num_layers=2).to(device) 
    #model = LSTM_Net2(input_dim=4, hidden_dim=100, batch_size=1,
                    #output_dim=2, num_layers=2).to(device) 

    if str(device) == 'cpu':
        #model.load_state_dict(torch.load('strategies/model_weights/model.pth', map_location=torch.device('cpu')))
        model.load_state_dict(torch.load('strategies/model_weights/model_lstm.pth', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('strategies/model_weights/model.pth'))

    print(f'\n{model}\n')

    wins = 0
    total = 0
    total_earn = 0
    y_pred = []
    y_true = []
    while iter > 0:
        '''
        iq.reconnect_after_10_minutes()
        utils.countdown()
        data = iq.get_candles()
        d = np.array(data).astype(np.float32)
        if d.shape[0] > 3:
            d = d[-3:]
        d = processx(d)
        d = MinMaxScaler(d)
        #pred = model.predict(d)[0]
        with torch.no_grad():
            outputs = model(torch.Tensor(d))
            _, predicted = torch.max(outputs.data, 1)
            pred = predicted.item()
        outputs = outputs[0]
        confs = F.softmax(outputs,0)
        print(f'confidence: | confs[pred] | {confs[0]:.4f}:{confs[1]:.4f} | {abs(confs[0]-confs[1]):.4f} | ', end='')
        print(f'{utils.now()} | {init_iter-iter:3d} | ', end='')
        if pred == 0: # trade normal
        #if pred == 1: # trade opposite
            print(f' put', end=' ')
            result, earn = iq.buy('put', check_result=True)
        else:
            print(f'call', end=' ')
            result, earn = iq.buy('call', check_result=True)
        if result == 'win':
            wins += 1
        if result != 'equal':
            if pred == 0:
                y_pred.append('put')
                if result == 'win':
                    y_true.append('put')
                else:
                    y_true.append('call')
            else:
                y_pred.append('call')
                if result == 'win':
                    y_true.append('call')
                else:
                    y_true.append('put')
        total += 1
        total_earn += earn
        print(f'| {result:>5} | ', end='')
        print(f'{str(round(earn,2)):>5} | ', end='')
        print(f'accuracy: {wins/total:.2f} | ', end='')
        print(f'total earn: {str(round(total_earn,2)):>6}')
        iter -= 1
        '''

        if 1:
            iq.reconnect_after_10_minutes()
            utils.countdown()
            data = iq.get_candles()
            d = np.array(data).astype(np.float32)
            if d.shape[0] > 3:
                d = d[-3:]
            d = processx(d)
            d = MinMaxScaler(d)
            #pred = model.predict(d)[0]
            with torch.no_grad():
                outputs = model(torch.Tensor(d))
                _, predicted = torch.max(outputs.data, 1)
                pred = predicted.item()
            confs = F.softmax(outputs[0],0)
            print(f'{utils.now()} | {init_iter-iter:3d} | ', end='')
            print(f'confidence: {confs[pred]:.6f} | ', end='')
            #if confs[pred] == 1:
                #iq.money = 10
            #else:
                #iq.money = 1
            if pred == 0: # trade normal
            #if pred == 1: # trade opposite
                print(f' put', end=' ')
                result, earn = iq.buy('put', check_result=True)
            else:
                print(f'call', end=' ')
                result, earn = iq.buy('call', check_result=True)
            if result == 'win':
                wins += 1
            if result != 'equal':
                if pred == 0:
                    y_pred.append('put')
                    if result == 'win':
                        y_true.append('put')
                    else:
                        y_true.append('call')
                else:
                    y_pred.append('call')
                    if result == 'win':
                        y_true.append('call')
                    else:
                        y_true.append('put')
            total += 1
            print(f'| {result:>5} | ', end='')
            print(f'{str(round(earn,2)):>6} | ', end='')
            print(f'accuracy: {wins/total:.2f} | ', end='')
            if display_total_earn:
                total_earn += earn
                print(f'total earn: {str(round(total_earn,2)):>6}')
            else:
                print()
            iter -= 1

        '''
        except TypeError:
            print('\n')
            cm = confusion_matrix(y_pred, y_true, labels=['put', 'call'], normalize='all')
            df = pd.DataFrame(cm, columns=['true_put', 'true_call'], index=['pred_put', 'pred_call'])
            final(iq, init_balance)
            sys.exit()
        except KeyboardInterrupt:
            print('\n')
            cm = confusion_matrix(y_pred, y_true, labels=['put', 'call'], normalize='all')
            df = pd.DataFrame(cm, columns=['true_put', 'true_call'], index=['pred_put', 'pred_call'])
            final(iq, init_balance)
            sys.exit()
        except:
            print('ERROR')
            pass
        '''

    print('\n')
    cm = confusion_matrix(y_pred, y_true, labels=['put', 'call'], normalize='all')
    df = pd.DataFrame(cm, columns=['true_put', 'true_call'], index=['pred_put', 'pred_call'])
    print()
    print(df)
    final(iq, init_balance)

def trail_stop(init_iter=2):
    import os
    import sys
    sys.path.append(os.getcwd())
    import utils
    from data import IQOption
    iq = IQOption(goal='EURUSD',
                  size=60,
                  maxdict=3,
                  money=1,
                  expiration_mode=1,
                  account='PRACTICE')
    init_balance = iq.get_balance()
    print('init_balance =', init_balance)

    c_iter = init_iter
    total_earn = 0
    result = init_balance
    while c_iter > 0:
        print(f'{utils.now()} | {init_iter-c_iter:3d} | ')
        iq.buy_forex('buy', trail_stop=True)
        iq.buy_forex('sell', trail_stop=True)
        while not iq.all_positions_closed_forex():
            time.sleep(1)
        time.sleep(10)
        result = iq.get_balance() - result
        total_earn += result
        print(f'| earn: {result:>5.2f} | ', end='')
        print(f'total earn: {str(round(total_earn,2)):>6}')
        c_iter -= 1
    final(iq, init_balance)

if __name__ == '__main__':
    #model = back_test()
    #run(model)
    #train()
    #run()
    fire.Fire()
