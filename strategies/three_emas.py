import numpy as np
import time
import datetime
import matplotlib.pyplot as ply
import talib

def get_sma(d, p):
    return np.average(d[-p:, 1])

def get_ema(d, p):
    sma = get_sma(d[:-1],p)
    part1 = d[-1,1] * (2/(1+p))
    part2 = sma * (1-(2/(1+p)))
    part3 = part1 + part2
    return part3

def get_action2(d):
    ma0 = talib.EMA(d[-500:,1],10)[-1]
    ma1 = talib.EMA(d[-500:,1],20)[-1]
    ma2 = talib.EMA(d[-500:,1],200)[-1]
    close = d[-1,1]
    if close > ma2 and ma0 > ma1:
        return 'call'
    elif close < ma2 and ma0 < ma1:
        return 'put'
    else:
        return None

def get_action(d):
    # 4, 9, 13
    ma0 = talib.EMA(d,4)
    ma1 = talib.EMA(d,9)
    ma2 = talib.EMA(d,13)
    data = [{'col':'Green','val':ma0},
            {'col':'Yellow','val':ma1},
            {'col':'Red','val':ma2},
            {'col':'Line','val':d[-1,1]}]
    data.sort(key=lambda a : a['val'], reverse=True)
    print([i['col'] for i in data])
    #print(f'{ma0:.5f} {ma1:.5f} {ma2:.5f}', end=' ')
    if ma0 > ma1 and ma1 > ma2 and d[-1,1] > ma0:
    #if ma0 > ma1:
        return 'call'
    elif ma0 < ma1 and ma1 < ma2 and d[-1,1] < ma0:
    #elif ma0 < ma1:
        return 'put'
    else:
        return None

def countdown():
    n = datetime.datetime.now().second
    while n != 58:
        print('counting down', n, end='\r')
        time.sleep(1)
        n = datetime.datetime.now().second

def run_forex_temp(iq):
    initial_balance = iq.get_balance()
    print('initial_balance =', initial_balance)
    iq.buy_forex('buy')
    time.sleep(20)
    iq.close_forex()
    terminal_balance = iq.get_balance()
    print('termnal_balance =', terminal_balance)
    print(round(terminal_balance - initial_balance,2))

def final(iq, initial_balance):
    while not iq.all_positions_closed_forex():
        if iq.num_open_positions == 1:
            print(f'{iq.num_open_positions} position is still open', end='\r')
        else:
            print(f'{iq.num_open_positions} positions are still open', end='\r')
        time.sleep(10)
    terminal_balance = iq.get_balance()
    max_len = max(len(str(initial_balance)), len(str(terminal_balance)))
    net_profit = terminal_balance - initial_balance
    print(' '*40)
    print(f' initial_balance: {initial_balance:>{max_len}}')
    print(f'terminal_balance: {terminal_balance:>{max_len}}')
    print(f'      net_profit: {net_profit:>{max_len}.2f} USD')
    print(f'      net_profit: {net_profit * 30.09:>{max_len}.2f} THB')

def get_action3(d):
    ma2 = talib.EMA(d[:,1],50)[-1]
    close = d[-1,1]
    #print('ma2:', round(ma2,5), '  close:', close)
    if close > ma2:
        return 'call'
    elif close < ma2:
        return 'put'
    else:
        return None

def run_forex2(iq):
    actions = {'put':'sell', 'call':'buy',
               'long':'buy','short':'sell',
               None:'None'}
    initial_balance = iq.get_balance()
    itt = 10
    iterations = itt
    while iterations > 0:
        iq.reconnect_after_30_minutes()
        data = np.array(iq.get_candles())
        init_action = get_action3(data)
        action = init_action
        while action == init_action:
            data = np.array(iq.get_candles())
            action = get_action3(data)
            print('action:', action, ' - init_action:', init_action)
        print('iterations:', iterations, ' itt:', itt, ' nums:')
        if iterations != itt:
            iq.close_all_forex()
        iq.buy_forex(actions[action])
        iterations -= 1
        print('iter:', iterations)
    final(iq,initial_balance)

def run_forex(iq):
    #iterations = 3
    initial_balance = iq.get_balance()
    print()
    print('initial_balance =', initial_balance)
    print()
    actions = {'put':'sell', 'call':'buy',
               'long':'buy','short':'sell',
               None:'None'}
    b1 = None
    b2 = None
    current_action = None
    current_status = None
    info = {}
    profit = 0
    iterations = 10
    while iterations > 0:
        #countdown()
        iq.reconnect_after_30_minutes()
        data = np.array(iq.get_candles())
        #action = get_action(data)
        action = get_action2(data)
        info['newact'] = actions[action]
        '''
        print('current_action:',current_action,
              ' actions[action]:', actions[action],
              ' status:', current_status)
        '''
        if action == None:
            current_action = None
            time.sleep(2)
            continue
        # ===
        else:
            if iq.all_positions_closed_forex():
                iq.buy_forex(actions[action])
                current_action = action
                iterations -= 1
                print('iterations:', iterations)
            else:
                if action != current_action:
                    iq.close_all_forex()

        # ===
        #===
        '''
        elif current_action != actions[action]:
            iterations -= 1
            if current_status == 'open':
                iq.close_forex()
            else:
                b1 = iq.get_balance()
            b2 = iq.get_balance()
            profit += b2 - b1
            b1 = b2
            iq.buy_forex(actions[action])
        elif current_action == actions[action]:
            if current_status == 'closed':
                iterations -= 1
                b2 = iq.get_balance()
                profit += b2-b1
                b1 = b2
                iq.buy_forex(actions[action])

        current_action, current_status = iq.get_position_forex()
        current_action = actions[current_action]
        info['action'] = current_action
        info['status'] = current_status
        info['P/L'] = round(profit,2)
        info['iter'] = iterations
        for k,v in info.items():
            print(f'{k}: {str(v):>6}', end='  |  ')
        print()
        '''
        #===
        '''
        if action == 'call':
            if current_action == 'short':
                iq.close_forex()
                iq.buy_forex('buy')
                iterations -= 1
            elif current_action is None or current_status == 'closed':
                iq.buy_forex('buy')
                iterations -= 1
        elif action == 'put':
            if current_action == 'long':
                iq.close_forex()
                iq.buy_forex('sell')
                iterations -= 1
            elif current_action is None or current_status == 'closed':
                iq.buy_forex('sell')
                iterations -= 1
        current_action, current_status = iq.get_position_forex()
        info['action'] = current_action
        info['P/L'] = round(iq.get_balance() - initial_balance,2)
        for k,v in info.items():
            print(k,':',v, end=' ')
        print()
        '''

    #iq.close_forex()
    while not iq.all_positions_closed_forex():
        if iq.num_open_positions == 1:
            print(f'{iq.num_open_positions} position is still open', end='\r')
        else:
            print(f'{iq.num_open_positions} positions are still open', end='\r')
        time.sleep(10)
    terminal_balance = iq.get_balance()
    max_len = max(len(str(initial_balance)), len(str(terminal_balance)))
    net_profit = terminal_balance - initial_balance
    print(' '*40)
    print(f' initial_balance: {initial_balance:>{max_len}}')
    print(f'terminal_balance: {terminal_balance:>{max_len}}')
    print(f'      net_profit: {net_profit:>{max_len}.2f} USD')
    print(f'      net_profit: {net_profit * 30.09:>{max_len}.2f} THB')

def run(iq, expiration_mode):
    iterations = 3
    initial_balance = iq.get_balance()
    print('initial_balance =', initial_balance)
    while iterations > 0:
        countdown()
        data = iq.get_candles()
        data = np.array(data)
        action = get_action(data)
        if action is not None:
            print(f'action: {action:>{4}} balance: {iq.get_balance()-initial_balance:.2f}')
            iq.buy(action, check_result=False)
            iterations -= 1
        else:
            time.sleep(60*expiration_mode)

    time.sleep(60 * expiration_mode)
    terminal_balance = iq.get_balance()
    max_len = max(len(str(initial_balance)), len(str(terminal_balance)))
    net_profit = terminal_balance - initial_balance
    print()
    print(f' initial_balance: {initial_balance:>{max_len}}')
    print(f'terminal_balance: {terminal_balance:>{max_len}}')
    print(f'      net_profit: {net_profit:>{max_len}.2f} USD')
    print(f'      net_profit: {net_profit * 30.09:>{max_len}.2f} THB')

def back_test():
    import os, sys
    sys.path.append(os.getcwd())
    for _, i in enumerate(sys.path):
        print(_,i)
    from practice_data import practice_data
    ptd = practice_data()
    all_data = ptd.all_data
    num_data = 50
    for data in all_data:
        actions = []
        for idx in range(num_data,data.shape[0]):
            #print(data[idx-num_data:idx].shape)
            d = data[idx-num_data:idx]
            action = get_action3(d)
            #print(action)
            actions.append(action)
        print(set(actions))
        break

if __name__ == '__main__':
    back_test()
