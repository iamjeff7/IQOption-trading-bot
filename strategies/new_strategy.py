import talib
import tqdm
import numpy as np
import matplotlib.pyplot as plt

'''
current_action = {'buy', 'sell', 'start', 'close',
                  'wait to buy', 'wait to sell'}
'''
def get_action(d, current_action, period):
    close = d[-1,1]
    close2 = d[-2,1]
    if current_action == 'start':
        ema = talib.EMA(d[:,1],period)[-5:]
        up = (ema[-1] - ema[0])/5 > 0
        if close > ema[-1] and up:
            return 'buy'
        elif close < ema[-1] and not up:
            return 'sell'
        else:
            return 'close'

    ema = talib.EMA(d[:,1],period)[-1]
    print(' '*80,end='\r')
    print(f'close: {close:.5f} - ema: {ema:.5f}')

    if current_action == 'sell':
        ubb, _, dbb = talib.BBANDS(d[:,1],14)
        ubb, dbb = ubb[-1], dbb[-1]
        if (close < dbb or close2 < dbb) or (close > ema):
            return 'close'
        else:
            return 'sell'
        '''
        if close > ema and close2 > ema:
            return 'close'
        else:
            return 'sell'
        '''
    elif current_action == 'buy':
        ubb, _, dbb = talib.BBANDS(d[:,1],14)
        ubb, dbb = ubb[-1], dbb[-1]
        if (close > ubb or close2 > ubb) or (close < ema):
            return 'close'
        else:
            return 'buy'
        '''
        if close > ema and close2 > ema:
            return 'buy'
        else:
            return 'close'
        '''

    elif current_action == 'close':
        if close > ema:
            return 'wait to sell'
        else:
            return 'wait to buy'

    elif current_action == 'wait to buy':
        up = close - close2 > 0
        if close > ema and close2 > ema and up:
            return 'buy'
        else:
            if close < ema and close2 < ema and not up:
                return 'wait to sell'
            return 'wait to buy'
    elif current_action == 'wait to sell':
        down = close - close2 < 0
        if close < ema and close2 < ema and down:
            return 'sell'
        else:
            if close > ema and close2 > ema and not down:
                return 'wait to buy'
            return 'wait to sell'

def back_test():
    import os, sys
    sys.path.append(os.getcwd())
    from practice_data import practice_data
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import clear_output

    ptd = practice_data()
    all_data = ptd.all_data
    period = 300
    money = 1000


    current_action = 'start'
    total_profit = 0
    last_price = 0
    ppp = []
    for didx, data in enumerate(all_data):
        current_action = 'start'
        actions = ['start']
        c = 0
        x_lin = []
        x_lin = []
        y_lin = []
        x_buy = []
        y_buy = []
        x_sell = []
        y_sell = []
        y_lin = []
        x_buy = []
        y_buy = []
        x_sell = []
        y_sell = []
        x_close = []
        y_close = []
        x_ema = []
        y_ema = []
        profit = 0
        open_price = 0
        pp = []

        for idx in range(period,data.shape[0]):
            d = data[idx-period:idx]
            action = get_action(d, current_action, period)
            if action == 'buy' or action == 'sell':
                open_price = d[-1,1]
                c += 1
            elif action == 'close' and c > 0:
                if current_action == 'buy':
                    profit += (d[-1,1] / open_price) - 1
                    #if d[-1,1] > open_price:
                        #profit += d[-1,1] - open_price
                    #else:
                        #profit += open_price - d[-1,1]
                elif current_action == 'sell':
                    profit += (open_price / d[-1,1]) - 1
                    #if d[-1,1] > open_price:
                        #profit += open_price - d[-1,1]
                    #else:
                        #profit += d[-1,1] - open_price
                pp.append(profit)
                ppp.append(profit)

            #fig, ax = plt.subplots(figsize=(13,8))

            ''' line '''
            if len(x_lin) == 0:
                y_lin = list(d[:,1])
                x_lin = [i for i in range(idx-period,idx)]
            else:
                x_lin.append(idx)
                y_lin.append(d[-1,1])
            ''' scatter '''
            if action == 'buy' and current_action == 'wait to buy':
                x_buy.append(idx)
                y_buy.append(d[-1,1])
            elif action == 'sell' and current_action == 'wait to sell':
                x_sell.append(idx)
                y_sell.append(d[-1,1])
            elif action == 'close':
                x_close.append(idx)
                y_close.append(d[-1,1])
            #ax.scatter(x_buy,y_buy,c='g',marker=6)
            #ax.scatter(x_sell,y_sell,c='r',marker=7)
            #ax.scatter(x_close,y_close,c='black',marker='x')

            ''' ema '''
            x_ema.append(idx)
            y_ema.append(talib.EMA(d[:,1],period)[-1])
            #ax.plot(x_ema, y_ema)


            c += 1
            #ly = np.arange(idx-period,idx)
            #ax.plot(ly,d[:,1], c='orange')
            #plt.title(current_action)
            #plt.draw()
            #plt.pause(0.001)
            #plt.close()

            #if current_action == 'close':
                #a = 3
            #a -= 1
            #print('action:',action, '- last_action:', current_action)
            #if a == 0:
                #plt.pause(20)
            #if len(x_buy)>0 and x_buy[0] < (idx-period):
            #    x_buy = x_buy[1:]
            #    y_buy = y_buy[1:]
            #if len(x_sell)>0 and x_sell[0] < (idx-period):
            #    x_sell = x_sell[1:]
            #    y_sell = y_sell[1:]
            #if len(x_close)>0 and x_close[0] < (idx-period):
            #    x_close = x_close[1:]
            #    y_close = y_close[1:]
            #x_ema = x_ema[-period:]
            #y_ema = y_ema[-period:]
            #last_price = d[-1,1]


            if c==3000:
                break
            #temp_act = current_action
            current_action = action

        fig, ax = plt.subplots(figsize=(13,8))
        ax.plot(x_ema, y_ema)
        ax.plot(x_lin, y_lin, c='orange', zorder=-1)
        ax.scatter(x_lin, y_lin, c='orange', s=5)
        ax.scatter(x_buy,y_buy,c='g',marker=6)
        ax.scatter(x_sell,y_sell,c='r',marker=7)
        ax.scatter(x_close,y_close,c='black',marker='x')
        plt.title(str(didx+1)+'/'+str(len(all_data)))
        plt.draw()
        plt.pause(5)
        #plt.show()
        plt.close()
        if current_action == 'open' or current_action == 'close':
            profit += last_price
        total_profit += profit
        print(f'Profit: {money*profit:.7f}')

        '''
        fig, ax = plt.subplots(figsize=(13,8))
        ax.plot(pp)
        #plt.show()
        plt.title(current_action)
        plt.draw()
        plt.pause(1)
        plt.close()
        '''
    print(f'Total Profit: {money*total_profit:.7f}')
    #fig, ax = plt.subplots(figsize=(13,8))
    #ax.plot(ppp)
    #plt.show()

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

def run_forex(iq):
    initial_balance = iq.get_balance()
    print()
    print('initial_balance =', initial_balance)
    print()

    current_action = 'start'
    actions = ['start']
    c = 0
    x_lin = []
    y_lin = []
    x_buy = []
    y_buy = []
    x_sell = []
    y_sell = []
    x_close = []
    y_close = []
    x_ema = []
    y_ema = []
    profit = 0
    period = 50
    idx = period
    money = 100

    init_iter = 30
    iterations = init_iter
    print(iterations, '/',init_iter)
    while iterations > 0:
        iq.reconnect_after_30_minutes()
        d = np.array(iq.get_candles())
        idx += 1
        action = get_action(d, current_action, period)
        print(' '*80,end='\r')
        print('  new action:',action, '- current action', current_action, end='\r')
        ''''''
        if (action == 'buy' or action == 'sell') and iq.all_positions_closed_forex():
            iq.buy_forex(action)
            open_price = d[-1,1]
            c += 1
        elif action == 'close' and c > 0:
            iq.close_all_forex()
            iterations -= 1
            print(' '*80)
            print(iterations, '/',init_iter)
            if current_action == 'buy':
                profit += (d[-1,1] / open_price) - 1
            elif current_action == 'sell':
                profit += (open_price / d[-1,1]) - 1

        ''' line '''
        #if len(x_lin) == 0:
            #y_lin = list(d[-period:,1])
            #x_lin = [i for i in range(idx-period,idx)]
        #else:
        x_lin.append(idx)
        y_lin.append(d[-1,1])

        ''' scatter '''
        if action == 'buy' and current_action == 'wait to buy':
            x_buy.append(idx)
            y_buy.append(d[-1,1])
        elif action == 'sell' and current_action == 'wait to sell':
            x_sell.append(idx)
            y_sell.append(d[-1,1])
        elif action == 'close':
            x_close.append(idx)
            y_close.append(d[-1,1])

        ''' ema '''
        x_ema.append(idx)
        y_ema.append(talib.EMA(d[:,1],period)[-1])

        current_action = action

    print()
    print(f'Profit: {money*profit:.7f}')
    print()
    final(iq, initial_balance)

    fig, ax = plt.subplots(figsize=(13,8))
    ax.plot(x_ema, y_ema)
    ax.plot(x_lin, y_lin, c='orange', zorder=-1)
    ax.scatter(x_lin, y_lin, c='orange', s=5)
    ax.scatter(x_buy,y_buy,c='g',marker=6)
    ax.scatter(x_sell,y_sell,c='r',marker=7)
    ax.scatter(x_close,y_close,c='black',marker='x')
    #plt.title(str(didx+1)+'/'+str(len(all_data)))
    #plt.draw()
    #plt.pause(5)
    plt.show()
    plt.close()


if __name__ == '__main__':
    back_test()

