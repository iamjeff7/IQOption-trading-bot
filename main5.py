from data import IQOption
from strategies import simple_ema
from strategies import price_action
import time
import numpy as np
import tqdm
import follow_mood
from strategies import three_emas
from strategies import new_strategy

if __name__ ==  '__main__':
    expiration_mode = 3
    iq = IQOption(goal='EURUSD',
                  size=60,
                  maxdict=1001,
                  money=1,
                  expiration_mode=expiration_mode,
                  account='PRACTICE')

    #iq.buy('call')
    # iq.buy(action)
    # action = simple_ema.get_action(data)

    # price_action
    # price_action.run(iq=iq, expiration_mode=expiration_mode)
    # iq.buy_forex()
    # follow_mood.run(iq=iq, expiration_mode=expiration_mode)
    # three_emas.run(iq, expiration_mode)
    # three_emas.run_forex(iq)
    # three_emas.run_forex2(iq)
    # three_emas.run_forex_temp(iq)
    new_strategy.run_forex(iq)

