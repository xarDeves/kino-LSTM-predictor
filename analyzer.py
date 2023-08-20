from operator import itemgetter
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import numpy as np
import os

def fetchFilenames():
    l = []
    for filename in os.listdir():
        if filename[-2:] != "py" :
            l.append(filename)
    return l

def printMinMax(data):
    _max = max(data,key=itemgetter(1))
    print('num {} is max with: {}'.format(_max[0], _max[1]))
    _min = min(data,key=itemgetter(1))
    print('num {} is min with: {}'.format(_min[0], _min[1]))
    #for x in sorted(merged, key=lambda tup: tup[1]):
    #    print(x)

def plotVals(data):
    plt.xticks(range(1, 81))
    plt.plot(*zip(*data))
    plt.plot(*zip(*data), 'or')
    plt.show()
    
def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list
     
def accumulateFile(filename):
    vals = np.array([*range(1,81)])
    counts = np.array([0] * 81)

    df = pd.read_excel('{}'.format(filename), engine='openpyxl')
    df = df.iloc[:, 3 : 23]

    c = 0
    for col in df:
        for val in (df[col]):
            try:
                counts[int(val) - 1] += 1
                c += 1
            except: pass
    merged = merge(vals, counts)
    merged.append(int(c / 2))
    return  merged

if __name__ == '__main__':
    vals = np.array([*range(1,81)], dtype=np.int64)
    counts = np.array([0] * 81)

    raffleCount = 0
    final = merge(vals, counts)
    
    filenames = fetchFilenames()

    with mp.Pool() as pool:
        results = pool.map(accumulateFile, filenames)
        for result in results:
            raffleCount += result[len(result) - 1]
            for i in range(len(result) - 1):
                final[i] = (final[i][0], final[i][1] + result[i][1])
                
    print('files processed: {}'.format(len(filenames)))
    print('raffles processed: {}'.format(raffleCount))
    printMinMax(final)
    plotVals(final)
