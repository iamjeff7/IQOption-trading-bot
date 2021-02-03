import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

'''
#a = np.arange(2*3*4).reshape(2,3,4)
a = np.random.randint(0,10,(2,3,4))
print(a)
# print(a[:,:,0])
# print(a[:,:,-1])
print('-------')
#a[:,:,0] = a[:,:,0] - a[:,:,-1]
for i in range(a.shape[2]):
    a[:, :, i] = a[:, :, -1] - a[:, :, i]
print(a)
'''

# a = pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
# a = pd.DataFrame({'a':[1,2,3,4]})
# print(a)
# print(a.describe())
# print(type(a.describe()))
# v = float(a.describe().loc['75%'])
# print(v, type(v))

# a = np.arange(20).reshape(4,5)
# b = np.ones(4).reshape(-1,1)
# print(a)
# a = np.hstack([a,b])
# print(a)
# print(a[a[:,-1]%2==0])
# np.random.shuffle(a)
# print(a)

# import torch as T
# #batch, #feature, #data
# def minmaxscaler(x):
#     # dims = batch, feature, data
#     return x
# a = np.random.randint(0,500,(3,2,5))
# a = T.tensor(a).to(T.float32)
# b = a
# a[0,0,0:3] = 1
# a[0,0,-1] = 20
# print(a)
# mins = T.min(a, dim=2, keepdim=True)[0]
# maxs = T.max(a, dim=2, keepdim=True)[0]
# a -= mins
# a = a/(maxs-mins)
# print(a)
# # sys.exit()
# mn,mx = T.min(b).item(),T.max(b).item()
# diff = mx-mn
# b -= mn
# b = b/diff
# # print(a)
# # print()
# fig, ax = plt.subplots(3, 4, figsize=(10,8),sharex=True, sharey=False)
# for r in range(3):
#     for c in range(2):
#         # print(r,c, a[r, c, :])
#         ax[r, c].plot(a[r, c, :])
# for r in range(3):
#     for c in range(2,4):
#         print(r, c, b[r, c-2, :])
#         ax[r, c].plot(b[r, c-2, :])
# plt.show()


# df = pd.DataFrame(np.random.randint(10,50,20),columns=['close'])
# print(df)
#
# n = 3
# for i in range(1, n + 1):
#     df['close_' + str(i)] = df['close'].shift(-i)
# period = 2 + n
# y = df['close'].shift(periods=-period)
# df['y'] = (df['close_'+str(n)] < y).astype(np.int32) # call = 1, put = 0
# pip = 15
# y2 = abs(df['close_'+str(n)] - y) < pip
# df.loc[y2, 'y'] = 2
# print(df[:-period])


# y_true = np.random.randint(low=0, high=3, size=1000, dtype=np.int)
# y_pred = np.random.randint(low=0, high=3, size=1000, dtype=np.int)
# classes_ = ['put','call','hold']
# u, count = np.unique(y_true, return_counts=True)
# print(u, count)
# u, count = np.unique(y_pred, return_counts=True)
# print(u, count)
#
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()

a = np.random.rand(5).astype(np.float64)
b = np.random.rand(5).astype(np.float64)
print(a)
print(b)

a = a[-1]
b = b[-1]
print(a-b)