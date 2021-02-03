import tensorflow as tf

#from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from tensorflow.keras import Model

import os, sys
import numpy as np
import pandas as pd

def process(data):
    df = pd.DataFrame(data, columns=['open','close','low','high'])
    n = 10
    for i in range(1,n+1):
        df['open_'+str(i)] = df['open'].shift(-i)
        df['close_'+str(i)] = df['close'].shift(-i)
        df['low_'+str(i)] = df['low'].shift(-i)
        df['high_'+str(i)] = df['high'].shift(-i)
    period = 30 + n
    df['y'] = df['close'].shift(periods=-period)
    df = df[df['close_'+str(n)]!=df['y']]
    df['y'] = (df['close_'+str(n)] < df['y']).astype(np.int32) # call = 1
    df = df.to_numpy()
    x,y = df[:,:-1], df[:,-1]
    return x,y
def one():
    filenames = [i for i in os.listdir() if 'npy' in i and not 'size' in i]
    data = np.load(filenames[0])
    x,y = process(data)
    print(y.shape)
    ratio = int(y.shape[0] * 0.75)
    print(ratio)

    x_train,x_test,y_train,y_test = x[:ratio],x[ratio:],y[:ratio],y[ratio:]

    train_ds = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


if __name__ == '__main__':
    one()
