import numpy as np
import time
import tqdm

def long_candle(d):
    d = d[-20:]
    high = np.max(d[:, 0:2], axis=1)
    low = np.min(d[:, 0:2], axis=1)
    longest_candle = np.max(high-low)
    thresh = longest_candle * 0.5
    c = d[-2].astype(np.float32)
    a = c[1]
    b = c[0]
    e = a-b
    print(f'{a:.5f} - {b:.5f} = {e:.5f} / {thresh:.5f}', end=' ')
    if e >= thresh:
        return 'call'
    elif e <= thresh*-1:
        return 'put'
    return None

def slope(d):
    d = d[-10:-2]
    y1, y2 = np.average(d[0,:2]), np.average(d[-1,:2])
    slope = (y2 - y1) / d.shape[0]
    slope = slope.astype(np.float32)
    print(f' {slope:.7}', end=' ')
    # [-1,1]
    if slope >= 0.00001:
        return 'call'
    elif slope <= -0.00001:
        return 'put'
    else:
        return None

def run(iq, expiration_mode):
    minutes = 60
    initial_balance = iq.get_balance()
    print('initial_balance =', initial_balance)
    while minutes > 0:
        data = iq.get_candles()
        action = long_candle(np.array(data))
        # action = slope(np.array(data))
        if action != None:
            print(':', action, end=' ')
            iq.buy(action, check_result=False)
        else:
            print(': pass', end=' ')
            time.sleep(1)
        print(f'{iq.get_balance()-initial_balance:.2f}')
        for i in tqdm.tqdm(range(60, 0, -1)):
            # print('waiting...',i, end='\r')
            time.sleep(1)
            pass
        minutes -= 1

    for i in tqdm.tqdm(range(60 * expiration_mode, 0, -1)):
        time.sleep(1)
    terminal_balance = iq.get_balance()
    max_len = max(len(str(initial_balance)), len(str(terminal_balance)))
    net_profit = terminal_balance - initial_balance
    print()
    print(f' initial_balance: {initial_balance:>{max_len}}')
    print(f'terminal_balance: {terminal_balance:>{max_len}}')
    print(f'      net_profit: {net_profit:>{max_len}.2f} USD')
    print(f'      net_profit: {net_profit * 30.09:>{max_len}.2f} THB')
    with open('log.txt', 'a') as f:
        # f.write('price_action_slope;')
        f.write('price_action;')
        f.write(str(round(net_profit * 30.09,2)))
        f.write(';THB;')
        import datetime
        cdt = datetime.datetime.now()
        f.write(str(cdt))
        f.write('\n')
