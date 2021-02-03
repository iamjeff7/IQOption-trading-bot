import numpy as np
import pandas as pd
import datetime

class IQOption:
    def __init__(self,goal,size,maxdict,money,expiration_mode, account='PRACTICE'):
        '''
        account : ['REAL', 'PRACTICE']
        '''
        import json
        from iqoptionapi.stable_api import IQ_Option
        data = json.load(open('credentials.json'))
        username = data['email']
        password = data['password']
        self.Iq = IQ_Option(username,password)
        print()
        print('logging in...')
        check, reason = self.Iq.connect()#connect to iqoption
        print('login:', 'SUCCESSFUL' if check else 'FAILED')
        print()
        assert check,True
        self.login_time = datetime.datetime.now()
        #self.Iq.reset_practice_balance()
        self.Iq.change_balance(account)
        ALL_Asset=self.Iq.get_all_open_time()
        if ALL_Asset["turbo"][goal]["open"]:
            goal = goal
        else:
            goal = goal+'-OTC'
        print('goal =', goal)

        self.goal = goal
        self.size = size
        self.maxdict = maxdict
        self.money = money
        self.expirations_mode = expiration_mode
        self.consecutive_error = 0
        self.forex_order_id = []
        instrument_type = 'forex'
        instrument_id = self.goal
        print('forex: leverage:',self.Iq.get_available_leverages(instrument_type, instrument_id))

        print()

        #self.test_forex()

    def buy(self,action, check_result):
        buy_success, buy_order_id = self.Iq.buy(self.money,self.goal,action,self.expirations_mode)
        if not check_result:
            return None, None
        if buy_success:
            result, earn = self.Iq.check_win_v4(buy_order_id)
            self.consecutive_error = 0
            return result, round(earn,2)
        else:
            print(action+' fail')
            self.consecutive_error += 1
            assert self.consecutive_error < 5, 'Failed more than 5 times'
            return None, None

    def get_candles(self):
        self.Iq.start_candles_stream(self.goal,self.size,self.maxdict)
        candles = self.Iq.get_realtime_candles(self.goal,self.size)
        self.Iq.stop_candles_stream(self.goal,self.size)

        candles = list(candles.values())
        d = [[c['open'],c['close'],c['min'],c['max']] for c in candles]
        data = pd.DataFrame(d, columns=['open','close','low','high'])
        #data = np.array(d)
        return data
    def get_balance(self):
        # current_balance = {'request_id': '', 'name': 'balances',
        # 'msg': [
        #   {'id': 414500451, 'user_id': 84068869, 'type': 1, 'amount': 0, 'enrolled_amount': 0, 'enrolled_sum_amount': 0, 'hold_amount': 0, 'orders_amount': 0, 'auth_amount': 0, 'equivalent': 0, 'currency': 'USD', 'tournament_id': None, 'tournament_name': None, 'is_fiat': True, 'is_marginal': False, 'has_deposits': False},
        #   {'id': 414500452, 'user_id': 84068869, 'type': 4, 'amount': 15023.81, 'enrolled_amount': 15023.811818, 'enrolled_sum_amount': 15023.811818, 'hold_amount': 0, 'orders_amount': 0, 'auth_amount': 0, 'equivalent': 0, 'currency': 'USD', 'tournament_id': None, 'tournament_name': None, 'is_fiat': True, 'is_marginal': False, 'has_deposits': False},
        #   {'id': 414500453, 'user_id': 84068869, 'type': 5, 'amount': 0, 'enrolled_amount': 0, 'enrolled_sum_amount': 0, 'hold_amount': 0, 'orders_amount': 0, 'auth_amount': 0, 'equivalent': 0, 'currency': 'BTC', 'tournament_id': None, 'tournament_name': None, 'is_fiat': False, 'is_marginal': False, 'has_deposits': False},
        #   {'id': 414500454, 'user_id': 84068869, 'type': 5, 'amount': 0, 'enrolled_amount': 0, 'enrolled_sum_amount': 0, 'hold_amount': 0, 'orders_amount': 0, 'auth_amount': 0, 'equivalent': 0, 'currency': 'ETH', 'tournament_id': None, 'tournament_name': None, 'is_fiat': False, 'is_marginal': False, 'has_deposits': False}
        # ], 'status': 0}
        # return self.Iq.get_balances()['msg'][1]['amount']
        return self.Iq.get_balance()

    def buy_forex(self, action):
        instrument_type = 'forex'
        instrument_id = self.goal
        side = action
        amount = 100
        leverage = 50
        type = 'market'
        limit_price = None
        stop_price = None
        stop_lose_kind = None
        stop_lose_value = None
        take_profit_kind = None
        take_profit_value = None
        #stop_lose_kind = 'percent'
        #stop_lose_value = 1.0
        #take_profit_kind = 'percent'
        #take_profit_value = 1.5
        use_trail_stop = False
        auto_margin_call = False
        use_token_for_commission = False
        check, order_id = self.Iq.buy_order(instrument_type=instrument_type, instrument_id=instrument_id,
                                                 side=side, amount=amount, leverage=leverage,
                                                 type=type, limit_price=limit_price, stop_price=stop_price,
                                                 stop_lose_value=stop_lose_value, stop_lose_kind=stop_lose_kind,
                                                 take_profit_value=take_profit_value, take_profit_kind=take_profit_kind,
                                                 use_trail_stop=use_trail_stop, auto_margin_call=auto_margin_call,
                                                 use_token_for_commission=use_token_for_commission)

        self.forex_order_id.append(order_id)
        '''
        print('- '*10)
        print(self.Iq.get_order(order_id))
        print('- '*10)
        print(self.Iq.get_positions(instrument_type))
        print('- '*10)
        print(self.Iq.get_position_history(instrument_type))
        print('- '*10)
        print(self.Iq.get_available_leverages(instrument_type, instrument_id))
        print('- '*10)
        import time
        time.sleep(10)
        print(self.Iq.close_position(order_id))
        print('- '*10)
        print(self.Iq.get_overnight_fee(instrument_type, instrument_id))
        print('- '*10)
        '''
    def all_positions_closed_forex(self):
        all_close = True
        self.num_open_positions = 0
        for order_id in self.forex_order_id:
            order_type, order_status = self.get_position_forex(order_id)
            if order_status == 'open':
                self.num_open_positions += 1
                all_close = False
        return all_close
    def close_forex(self):
        self.Iq.close_position(self.forex_order_id)
    def close_all_forex(self):
        for order_id in self.forex_order_id:
            self.Iq.close_position(order_id)
    def get_position_forex(self, order_id):
        #return self.Iq.get_position(self.forex_order_id)[1]['position']['status']
        order = self.Iq.get_position(order_id)[1]['position']
        order_type = order['type']
        order_status = order['status']
        return order_type, order_status
    def test_forex(self):
        import time
        print('= '*20)
        self.buy_forex('sell')
        print(self.get_position_forex())
        print()
        time.sleep(5)
        self.close_forex()
        print(self.get_position_forex())
        print()
        print('= '*20)
        print()
        self.buy_forex('buy')
        print(self.get_position_forex())
        print()
        time.sleep(5)
        self.close_forex()
        print(self.get_position_forex())
        print()
        print('= '*20)
    def reconnect_after_30_minutes(self):
        b = datetime.datetime.now()
        #print((b-self.login_time).seconds, 'seconds')
        if (b-self.login_time).seconds > 60 * 30:
            check, reason = self.Iq.connect()#connect to iqoption
            assert check,True
            print(f'reconnected after {(b-self.login_time).seconds} seconds!')
            self.login_time = datetime.datetime.now()
