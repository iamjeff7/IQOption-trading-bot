import datetime
from iqoptionapi.stable_api import IQ_Option
import time
import numpy as np


def r(x):
    return round(x, 8)


print('\n\n')
'''
account : ['REAL', 'PRACTICE']
'''
import json
from iqoptionapi.stable_api import IQ_Option
data = json.load(open('credentials.json'))
username = data['email']
password = data['password']
Iq = IQ_Option(username,password)
print('logging in...')
check, reason = Iq.connect()#connect to iqoption
print('login:', 'SUCCESSFUL' if check else 'FAILED')
assert check

goal = 'EURUSD'
ALL_Asset = Iq.get_all_open_time()
if ALL_Asset["turbo"][goal]["open"]:
    goal = goal
else:
    goal = goal+'-OTC'
print()
print('Goal =', goal)
size = 60 * 60 * 4
maxdict = 10000
start = time.time()
Iq.start_candles_stream(goal, size, maxdict)
candles = Iq.get_realtime_candles(goal, size)
Iq.stop_candles_stream(goal, size)
end = time.time()
print()
print('len:', len(list(candles.values())))

candles = list(candles.values())
print(candles[0])

d = [[r(c['open']), r(c['close']), r(c['min']), r(c['max'])] for c in candles]

for idx, dd in enumerate(d):
    print(dd)
    if idx == 10:
        break

current_date_and_time = datetime.datetime.now()
print(current_date_and_time)

hours = 0
hours_added = datetime.timedelta(hours=hours)

units = ['seconds', 'minute', 'hour', 'day']
uidx = 0
while size > 59:
    size /= 60
    uidx += 1

fdt = str(current_date_and_time + hours_added)[:-7]
fdt = fdt.replace(' ', '_').replace('-', '').replace(':','-')
filename = 'dataset/'+goal+'_'+str(maxdict)+'_'+str(round(size))+'_'+units[uidx]+'_'+fdt
print(f'it took {end-start:.2f} seconds to get data')
print(filename+'\n')
np.save(filename, np.array(d))
print('DONE')
