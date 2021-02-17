'''
bollinger bands 14, 2
CCI 14
Stochastic Ocillator 13, 3, 3
'''

import talib
import numpy as np
import fire
import pandas as pd
import sys
import utils

from sklearn.metrics import confusion_matrix

def get_action(d):
    d = d.astype(np.float64)
    close, high, low = d[:, 1], d[:, 2], d[:, 3]
    upper, middle, lower = talib.BBANDS(close, timeperiod=14)
    real = talib.CCI(high, low, close, timeperiod=14)
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    #print(f' close:', close[-2:])
    #print(f'BBANDS: {upper[-2:]} {lower[-2:]}')
    #print(f'   CCI: {real[-2:]}')
    #print(f' STOCH: {slowk[-2:]} {slowd[-2:]}')
    #print(close[-1] > upper[-1], real[-1] > 100, slowk[-1] > 80, slowd[-1] > 80)
    #print(close[-1] < lower[-1], real[-1] < -100, slowk[-1] < 20, slowd[-1] < 20)
    if close[-2] > upper[-2] and real[-2] > 100 and slowk[-2] > 80 and slowd[-2] > 80:
        if close[-2] < close[-1]:
            return 'call'
        else:
            return 'put'
    elif close[-2] < lower[-2] and real[-2] < -100 and slowk[-2] < 20 and slowd[-2] < 20:
        if close[-2] < close[-1]:
            return 'call'
        else:
            return 'put'
    else:
        return None

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

def run(init_itr, display_total_earn):
    init_itr = int(init_itr)
    display_total_earn = display_total_earn.lower() in ['true','yes','y']
    import os
    import sys
    sys.path.append(os.getcwd())
    import utils
    from data import IQOption
    iq = IQOption(goal='EURUSD',
                  size=15,
                  maxdict=20,
                  money=1,
                  expiration_mode=2,
                  account='PRACTICE')
    init_balance = iq.get_balance()
    print('init_balance =', init_balance)

    wins = 0
    total = 0
    total_earn = 0
    y_pred = []
    y_true = []
    #init_itr = 3
    itr = init_itr
    while itr > 0:
        #if 1:
        try:
            iq.reconnect_after_10_minutes()
            #utils.countdown()
            data = iq.get_candles()
            d = np.array(data).astype(np.float32)
            if d.shape[0] > 20:
                d = d[-20:]
            pred = get_action(d)
            #print(f'{utils.now()} | {pred} |')

            if pred == 'put': # trade normal
            #if pred == 'call': # trade opposite
                print(f' put |', end=' ')
                result, earn = iq.buy('put', check_result=True)
            elif pred == 'call':
                print(f'call |', end=' ')
                result, earn = iq.buy('call', check_result=True)
            else:
                continue
            if result == 'win':
                wins += 1
            if result != 'equal':
                total += 1
                if pred == 'put':
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
            print(f'{utils.now()} | {init_itr-itr:3d} | ', end='')
            print(f'{result:>5} | ', end='')
            print(f'{str(round(earn,2)):>6} | ', end='')
            if total > 0:
                print(f'accuracy: {wins/total:.2f} | ', end='')
            else:
                print(f'accuracy: 0.00 | ', end='')
            if display_total_earn:
                total_earn += earn
                print(f'total earn: {str(round(total_earn,2)):>6}')
            else:
                print()
            itr -= 1

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
            print(f'\nERROR')
            sys.exit()

    print('\n')
    cm = confusion_matrix(y_pred, y_true, labels=['put', 'call'], normalize='all')
    df = pd.DataFrame(cm, columns=['true_put', 'true_call'], index=['pred_put', 'pred_call'])
    print()
    print(df)
    if init_itr >= 0:
        with open('strategies/katie_log.txt','a') as f:
            txt = f'{utils.now()} | {init_itr:>4} | acc: {wins/total:.2f}\n'
            print(txt)
            f.write(txt)
    final(iq, init_balance)


if __name__ == '__main__':
    fire.Fire()
