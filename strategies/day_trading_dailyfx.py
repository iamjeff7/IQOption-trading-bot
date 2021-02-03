import talib
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime


def get_action(d):
    #d = d[:-1]
    ema8 = talib.EMA(d[:, 1], 8)
    a = ema8[-1]
    b = ema8[-3]
    ema30 = talib.EMA(d[:, 1], 200)[-1]
    close = d[-1, 1]
    openn = d[-1, 0]
    middle = (openn+close)/2

    # if middle > ema30 and close < a and b < a:
    if middle > ema30 and close < a:
        #print(f'{close:.5f} : {b:.5f} {a:.5f}')
        return 'buy'
    # elif middle < ema30 and close > a and b > a:
    elif middle < ema30 and close > a:
        return 'sell'
    '''
    if middle > ema30:
        if close < ema8:
            return 'buy'
    else:
        if close > ema8:
            return 'sell'
    '''
    return None


def get_close(bought, d, mul, action, highest_profit):
    ema30 = talib.EMA(d[:, 1], 200)[-1]
    close = d[-1, 1]
    middle = (close + d[-1,0]) / 2
    pip = 0.0001 * mul
    diff = abs(bought - close)
    print(f'{now()} | {abs(bought-close):.5f} > {pip:.5f}? | {diff:.5f}/{highest_profit:.5f} | ', end='')
    cond1 = action == 'buy'and middle < ema30
    cond2 = action == 'sell' and middle > ema30
    cond3 = abs(bought - close) <= highest_profit*0.8
    if diff > pip or cond1 or cond2 or cond3:
        print('HELL YESSSS!')
        return 'close', 0
    else:
        print('no, not yet')
        if diff < highest_profit:
            return None, highest_profit
        else:
            return None, diff


def back_test(mul, a):
    print('Lets back test!')
    import os
    import sys
    sys.path.append(os.getcwd())
    from practice_data import PracticeData

    ptd = PracticeData()
    all_data = ptd.all_data
    period = 301

    all_win = 0
    all_total = 0
    all_loss = 0
    con_loss = 0
    rest = 0

    for didx, data in enumerate(all_data):
        x_lin = []
        y_lin = []
        x_buy = []
        y_buy = []
        x_sell = []
        y_sell = []
        x_close = []
        y_close = []
        x_ema0 = []
        y_ema0 = []
        x_ema1 = []
        y_ema1 = []
        x_rest = []
        y_rest = []
        x_up, y_up = [], []
        x_down, y_down = [], []
        bought = 0
        current_action = 'close'

        win = 0
        loss = 0
        total = 0
        con_loss = 0
        rest = 0

        for idx in range(period,data.shape[0]):
            d = data[idx-period:idx]
            action = None
            if current_action == 'close':
                action = get_action(d)
                if action is not None:
                    if rest != 0:
                        rest -= 1
                        x_rest.append(idx)
                        y_rest.append(d[-1, 1])
                        action = None
                    else:
                        bought = d[-1, 1]
                        current_action = action
            else:
                action = get_close(bought, d, mul)
                if action == 'close':
                    total += 1
                    if (current_action == 'buy' and bought < d[-1, 1]) or (current_action == 'sell' and bought > d[-1, 1]):
                        win += abs(bought - d[-1, 1]) / 0.0001
                        con_loss = 0
                    else:
                        loss += abs(bought - d[-1, 1]) / 0.0001
                        con_loss += 1
                        if con_loss == 3:
                            con_loss = 0
                            #rest = 3
                    total += 1
                    current_action = action

            ''' ema '''
            temp = talib.EMA(d[:, 1], 8)
            x_ema0.append(idx)
            y_ema0.append(talib.EMA(d[:, 1], 8)[-1])
            x_ema1.append(idx)
            y_ema1.append(talib.EMA(d[:, 1], 300)[-1])
            #if len(y_ema0) > 2:
                #print(f' 1 {temp[-2]:.5f} {temp[-1]:.5f}')
                #print(f' {len(y_ema0):4d} {y_ema0[-2]:.5f} {y_ema0[-1]:.5f}')
                #if y_ema0[-2] < y
            ''' line '''
            if len(x_lin) == 0:
                y_lin = list(d[:, 1])
                x_lin = [i for i in range(idx-period, idx)]
            else:
                x_lin.append(idx)
                y_lin.append(d[-1, 1])
            ''' scatter '''
            if action == 'buy':
                x_buy.append(idx)
                y_buy.append(d[-1, 1])
            elif action == 'sell':
                x_sell.append(idx)
                y_sell.append(d[-1, 1])
            elif action == 'close':
                x_close.append(idx)
                y_close.append(d[-1, 1])

        x_lin = x_lin[period:]
        y_lin = y_lin[period:]
        fig, ax = plt.subplots(figsize=(13,8))
        ax.plot(x_ema0, y_ema0, c='#00FF80')
        ax.plot(x_ema1, y_ema1, c='#E62200')
        ax.plot(x_lin, y_lin, c='orange', zorder=-1)
        ax.scatter(x_lin, y_lin, c='orange', s=5)
        ax.scatter(x_ema0, y_ema0, c='#00FF80', s=5)
        ax.scatter(x_buy, y_buy, c='g', marker=6)
        ax.scatter(x_sell, y_sell, c='r',marker=7)
        ax.scatter(x_close, y_close, c='black', marker='x')
        ax.scatter(x_rest, y_rest, c='gray',marker='.')
        plt.title(ptd.filenames[didx]+' '+str(didx+1)+'/'+str(len(all_data)))
        plt.close()

        #a = 2
        if a == 0:
            plt.draw()
            plt.pause(5)
        elif a == 1:
            plt.show()
        print()
        print(ptd.filenames[didx])
        print(f'  win: {win:.2f}')
        print(f' loss: {loss:.2f}')
        print(f' diff: {win-loss:.2f}')
        #print(f'total: {total}')
        #try:
            #print(f'  acc: {win/total:.2f}')
        #except:
            #print(f' acc: Not available')
        all_win += win
        all_loss += loss
        all_total += total
        #break

    print()
    print('='*40)
    print('final result')
    print(f'  win: {all_win:.2f}')
    print(f' loss: {all_loss:.2f}')
    print(f' diff: {all_win - all_loss:.2f}')
    return round(all_win - all_loss, 2)

def log(t):
    print(t, end='')
    with open(__file__.split('\\')[-1][:-3]+'-log.txt', 'a') as f:
        f.write(t)

def now():
    return str(datetime.now())[:-7]
def run_forex():
    import os
    import sys
    sys.path.append(os.getcwd())
    import utils
    from data import IQOption
    iq = IQOption(goal='EURUSD',
                  # size=60*60*4,
                  size=60,
                  maxdict=301,
                  money=1,
                  expiration_mode=1,
                  account='PRACTICE')
    initial_balance = iq.get_balance()
    print(f'\ninitial_balance = {initial_balance} \n')

    init_iter = 100
    iter = init_iter
    bought = 0
    current_action = 'close'
    win, loss = 0, 0
    highest_profit = 0

    log(f'\n\t{iter:>{len(str(init_iter))}}/{init_iter} {now()}\n')
    while iter > 0:
        iq.reconnect_after_30_minutes()
        t1 = time.time()
        d = np.array(iq.get_candles())
        # print(f'took {round(time.time()-t1,2)} seconds to get the candles')
        action = None
        if current_action == 'close':
            action = get_action(d)
            print(now(), '| ', action)
            if action is not None:
                bought = d[-1, 1]
                t1 = time.time()
                iq.buy_forex(action)
                print(f'took {round(time.time()-t1,2)} seconds to {action}')
                current_action = action
                log(f'{now()} | ')
                log(f'  {action}\n')
        else:
            action, highest_profit = get_close(bought, d, 24, current_action, highest_profit)
            if action == 'close':
                if (current_action == 'buy' and bought < d[-1, 1]) or (current_action == 'sell' and bought > d[-1, 1]):
                    win += abs(bought - d[-1, 1]) / 0.0001
                else:
                    loss += abs(bought - d[-1, 1]) / 0.0001
                current_action = action
                iter -= 1
                iq.close_all_forex()
                log(f'{now()} | ')
                log(f'  close : {iq.get_balance() - initial_balance:.2f}\n')
                log(f'\t{iter:>{len(str(init_iter))}}/{init_iter} {now()} \n')
                today_now = datetime.now()
                if today_now.day == 4 and today_now.hour > 12:
                    log(f'It\'s Friday. Let\'s take a break!\n')
                    break
    utils.final(iq, initial_balance, log)




if __name__ == '__main__':

    # profit = []
    # for mul in range(1,200):
    #     profit.append([mul, back_test(mul, 2)])
    # import pandas as pd
    # df = pd.DataFrame(profit, columns=['pip num', 'profit'])
    # df = df.sort_values('profit', ascending=False)
    # print(df.head(40))

    #back_test(1, 1)

    run_forex()
