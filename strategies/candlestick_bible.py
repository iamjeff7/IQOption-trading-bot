import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import fire

def get_high(data):
    data = data[:, 1]
    xy = []
    for idx in range(1, len(data)-1):
        a, b, c = idx-1, idx, idx+1
        if data[a] < data[b] > data[c]:
            xy.append([idx, data[b]])
    return np.array(xy)

def get_low(data):
    data = data[:, 1]
    xy = []
    for idx in range(1, len(data)-1):
        a, b, c = idx-1, idx, idx+1
        if data[a] > data[b] < data[c]:
            xy.append([idx, data[b]])
    return np.array(xy)

def candlestick(data, ax):
    '''
    bull_candle = []
    bear_candle = []
    for idx, d in enumerate(data):
        if d[0] < d[1]:
            bull_candle.append(idx)
        else:
            bear_candle.append(idx)
    ax.vlines(x=bull_candle, ymin=data[bull_candle,1],
            ymax=data[bull_candle,0], colors='green', lw=3)
    ax.vlines(x=bull_candle, ymin=data[bull_candle,3],
            ymax=data[bull_candle,2], colors='green', lw=1)

    ax.vlines(x=bear_candle, ymin=data[bear_candle,1],
            ymax=data[bear_candle,0], colors='red', lw=3)
    ax.vlines(x=bear_candle, ymin=data[bear_candle,3],
            ymax=data[bear_candle,2], colors='red', lw=1)
    '''

    lw = 1
    for idx, d in enumerate(data):
        if d[0] < d[1]:
            #bull_candle.append(idx)
            #print(f'open: {idx}, {d[0]:5f}, {d[1]-d[0]:.5f}')
            #ax.add_patch(Rectangle((idx,d[0]), 2, d[1]-d[0]))
            ax.add_patch(Rectangle((idx-lw/2,d[0]),lw,d[1]-d[0],
                edgecolor='green',
                facecolor='green'))
            ax.add_patch(Rectangle((idx,d[3]),0.1,d[2]-d[3],
                edgecolor='green',
                facecolor='green'))
        else:
            #bear_candle.append(idx)
            #ax.add_patch(Rectangle((idx,d[1]), 1, d[0]-d[1]))
            #print(f'close: {idx}, {d[0]:5f}, {d[0]-d[1]:.5f}')
            ax.add_patch(Rectangle((idx-lw/2,d[1]),lw,d[0]-d[1],
                edgecolor='red',
                facecolor='red'))
            ax.add_patch(Rectangle((idx,d[3]),0.1,d[2]-d[3],
                edgecolor='red',
                facecolor='red'))
    plt.show()
    

def back_test():
    import os
    import sys
    sys.path.append(os.getcwd())
    from practice_data import PracticeData2

    ptd = PracticeData2('EURUSD', 1)
    all_data = ptd.data
    sample = all_data[:10]
    high = get_high(sample)
    low = get_low(sample)

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(sample[:, 1], alpha=0)
    #ax.scatter(high[:, 0], high[:, 1], c='green')
    #ax.scatter(low[:, 0], low[:, 1], c='red')
    candlestick(sample, ax)
    

    plt.show()
    plt.close()


if __name__ == '__main__':
    fire.Fire()

