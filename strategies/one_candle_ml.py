import numpy as np
import os
import pandas as pd

def process_data(d):
    x = min_max_normalization(d)
    y = more_or_less(d)
    d = np.hstack([x,y.reshape(-1,1)])
    print('x:', x.shape)
    print('y:', y.shape)
    print('d:', d.shape)
    return d
def min_max_normalization(d):
    _d = d.astype(np.float32)
    mn = np.min(d, axis=1).reshape(-1,1)
    mx = np.max(d, axis=1).reshape(-1,1)
    mn -= 0.000001
    mx += 0.000001
    _d = (_d-mn) / (mx-mn)
    _d = d[:-1]
    return _d
def more_or_less(d):
    current_y = d[:-1,1]
    next_y = d[1:,1]
    # 0 = put, 1 = call
    _y = (current_y<next_y).astype(np.int32)
    return _y

def main():
    data = []
    for f in os.listdir('dataset'):
        f = os.path.join('dataset',f)
        d = np.load(f)
        d = process_data(d)
        data.append(d)
        break
    data = np.vstack(data)
    print(data[:5])
    np.random.shuffle(data)
    print(data[:5])
    print('data:', data.shape)
    a = np.unique(data[:,-1], return_counts=True)
    print(a)

if __name__ == '__main__':
    main()
