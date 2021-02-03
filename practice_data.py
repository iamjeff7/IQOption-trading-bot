import numpy as np
import os


class PracticeData:
    def __init__(self):
        self.filenames = []
        self.all_data = self._load_data()

    def _load_data(self):
        all_data = []
        for f in os.listdir('dataset'):
            if 'hour' not in f:
                continue
            self.filenames.append(f)
            f = 'dataset/'+f
            all_data.append(np.load(f))
        return all_data


if __name__ == '__main__':
    pdata = PracticeData()
    print(len(pdata.all_data))
    print(pdata.all_data[0].shape)
    import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2,7,figsize=(15,8), sharey=True)
    # for idx in range(len(pdata.all_data)):
    #     r,c = idx//7, idx%7
    #     axs[r,c].plot(pdata.all_data[idx])
    # plt.show()

    def long_bullish(d):
        idx = []
        val = []
        pip = 0.0001
        pip_mul = 3
        thresh = pip * pip_mul
        for i,dd in enumerate(d):
            if dd[1] - dd[0] > thresh:
                idx.append(i)
                val.append(dd)
        return idx, np.array(val)

    for idx in range(len(pdata.all_data[:5])):
        fig, axs = plt.subplots(figsize=(15, 8))
        axs.plot(pdata.all_data[idx][:,1])
        axs.scatter(np.arange(len(pdata.all_data[idx])),pdata.all_data[idx][:,0], c='#B5EAD7')
        axs.scatter(np.arange(len(pdata.all_data[idx])),pdata.all_data[idx][:,1], c='#FFDAC1')
        axs.scatter(np.arange(len(pdata.all_data[idx])),pdata.all_data[idx][:,2], c='#C7CEEA')
        axs.scatter(np.arange(len(pdata.all_data[idx])),pdata.all_data[idx][:,3], c='#FF9AA2')

        x,y = long_bullish(pdata.all_data[idx])
        if len(x) > 0:
            axs.scatter(x,y[:,2],c='red')
        plt.show()
