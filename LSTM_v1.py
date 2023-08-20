import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD

# time intervals in data set are 5 minutes
# 1 hour/5 mins is 12 entries
# look 4 hours back in time
WINDOW = 12 * 4


def plotPredictions(model, x, y, start=0, end=100, nums=20):
    predictions = model.predict(x)

    for i in range(nums):
        pred = predictions[:, i]
        actuals = y[:, i]
        df = pd.DataFrame(data={'Pred': pred, 'Act': actuals})
        plt.figure()
        plt.plot(df['Pred'][start:end], color='blue')
        plt.plot(df['Act'][start:end], color='orange')

    plt.show()

    # return df[start:end]


def fetchTrainingDataframe(filename):
    return pd.read_csv(filename, names=[
        # "day", "month", "year", "hour", "minute",
        "n1", "n2", "n3", "n4", "n5",
        "n6", "n7", "n8", "n9", "10",
        "n11", "n12", "n13", "n14", "n15",
        "n16", "n17", "n18", "n19", "20"
    ])


def normalizeTrainingSet(trainingSet):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(trainingSet)


def transformTrainingData(normalizedSet):
    x = []
    y = []

    for i in range(len(normalizedSet) - WINDOW):
        row = [x for x in normalizedSet[i:i + WINDOW]]
        x.append(row)
        label = [x for x in normalizedSet[i + WINDOW]]
        # label = [i + WINDOW]
        y.append(label)

    return np.array(x), np.array(y)


if __name__ == '__main__':
    trainingSet = fetchTrainingDataframe('NO DATES RAW 2022_10.csv')
    # trainingSet = fetchTrainingDataframe('NO DATES DOUBLE.csv')
    trainingSet = trainingSet.to_numpy()
    normalizedSet = normalizeTrainingSet(trainingSet)

    # X, Y = getTrainingData(trainingSet)
    X, Y = transformTrainingData(normalizedSet)
    # for NO DATES RAW 2022_10.csv
    trainX, trainY = X[:5300], Y[:5300]
    valX, valY = X[5300:5800], Y[5300:5800]
    testX, testY = X[5800:], Y[5800:]
    # for NO DATES DOUBLE.csv
    '''trainX, trainY = X[:9000], Y[:9000]
    valX, valY = X[9000:10000], Y[9000:10000]
    testX, testY = X[10000:], Y[10000:]'''
    print(trainX.shape, trainY.shape)
    print(valX.shape, valY.shape)
    print(testX.shape, testY.shape)

    model = Sequential()
    model.add(InputLayer((WINDOW, 20)))
    model.add(LSTM(20, dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(20, dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(20))
    model.add(BatchNormalization())
    # model.add(Dense(20, activation='relu'))
    # model.add(Dense(20, activation='softmax'))
    print(model.summary())

    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.00002),
        # optimizer=SGD(learning_rate=0.00002),
        metrics=[MeanAbsoluteError()]
        # metrics = [MeanSquaredError()]
    )

    # cp = ModelCheckpoint('model/', save_best_only=True)
    model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        epochs=200,
        batch_size=64
        # batch_size=16
        # ,callbacks=[cp]
    )

    plotPredictions(model, testX, testY, 0, 200)