import torch
import os
import matplotlib.pyplot as plt
import numpy as np

print(os.listdir('../'))
# x_train = torch.load('../x_train.pt')
x_test = torch.load('../x_test.pt')
y_test = torch.load('../y_test.pt')

# print('x_train =', x_train.size())
print('x_test =', x_test.size())

N = 4
for N in range(4,7):
    rid = np.random.randint(0,x_test.size(0),N)
    print(rid)

    x = x_test[[rid]]
    y = y_test[[rid]]

    fig, ax = plt.subplots(N, 4, figsize=(10,8),sharex=True, sharey=True)
    for r in range(N):
        for c in range(4):
            # print(r,c, a[r, c, :])
            ax[r, c].plot(x[r, c, :])
        ax[r, 0].set_title(y[r].item())
    plt.show()
