from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pca


if __name__ == '__main__':
    # test = [{'thing1': 5, 'thing2': 10, 'thing3': 15}]
    # test_pd = pd.DataFrame(test)
    #
    # test2 = [{'thing3': 25, 'thing1': 15, 'thing2': 20}]
    # test2_pd = pd.DataFrame(test2)
    #
    # test_pd = test_pd.append(test2_pd)
    # print("done")

    df1 = pd.read_csv("data/dataset.csv")
    df2 = pd.read_csv("data/christopher.csv")
    df3 = pd.read_csv("data/mason.csv")
    df = df1.append(df2).append(df3)
    cols = list(df.columns.values)
    cols.pop(cols.index("won?"))
    df = df[cols+["won?"]]

    init_data = pca.convert_data(df, 5)

    # init_data = df.to_numpy()

    # ## Normalize the data ###
    scaler = MinMaxScaler()
    scaler.fit(init_data)
    norm_data = scaler.transform(init_data)

    # data = norm_data[:, 0:-1]
    # labels = norm_data[:, -1]

    data = norm_data
    labels = df["won?"].to_numpy()

    train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=0.1)

    model = SVC(kernel='poly')  # , learning_rate_init=0.1, solver='sgd', momentum=0.5, activation='logistic',
    # early_stopping=True, alpha=0.0001)  # , hidden_layer_sizes=(48,))

    model.fit(train_X, train_y)
    print(round(model.score(train_X, train_y), 4), round(model.score(test_X, test_y), 4))
    # print("Training set accuracy:", round(model.score(train_X, train_y), 4))
    # print("Testing set accuracy:", round(model.score(test_X, test_y), 4))
