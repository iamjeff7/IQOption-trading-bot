import pandas as pd
import math

def get_sma(df, p):
    sma_str = 'sma'+str(p)
    temp = df['close'].rolling(p).mean()
    df[sma_str] = temp
    return df
def get_ema(df, p):
    temp1 = df['sma'+str(p)]
    # EMA
    ema_short = []
    prev = None
    ## EMA Short
    for v, c in zip(temp1, df['close']):
        if math.isnan(v):
            ema_short.append(v)
        else:
            if prev == None:
                ema_short.append(v)
                prev = v
            else:
                value = (c * (2/(1+p))) + (prev * (1-(2/(1+p))))
                prev = value
                ema_short.append(value)
    ema_short = pd.Series(ema_short)
    df['ema'+str(p)] = ema_short
    return df
def get_slope(df, t):
    temp = df[t].diff(2)
    df[t+' diff'] = round(temp,6)
    #df['sma'+str(p)+' diff'] = abs(round(temp,6))
    return df

def get_action(df):
    period = 200
    df = get_sma(df,period)
    df = get_ema(df,period)
    df = get_slope(df,'sma'+str(period))
    df = get_slope(df,'ema'+str(period))
    v = df.iloc[-2]['ema'+str(period)+' diff']
    action = None
    if v > 0 and v > 0.00001:
        action = 'call'
    elif v < 0 and v < -0.00001:
        action = 'put'
    '''
    else:
        print('None', end='\r')
        if n_action < loop-1:
            for _ in range(60,0,-1):
                time.sleep(1)
    '''
    return action
