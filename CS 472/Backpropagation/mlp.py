import numpy as np
from arff import Arff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# ## NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified.


class MLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=0.1, momentum=0, shuffle=True, hidden_layer_widths=None,
                 epochs=None, val_size=0.0, start_weights=None):
        """ Initialize class with chosen hyperparameters.
        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.epochs = epochs
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.val_size = val_size
        self.start_weights = start_weights
        self.weights = []
        self.num_classes = 0

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.num_classes = len(np.unique(y))
        self.weights = self.initialize_weights(len(X[0])) if not initial_weights else initial_weights
        # print("Baseline accuracy:", 100/self.num_classes)

        if self.epochs is None:
            X, y, val_x, val_y = self.split_data(X, y, (1-self.val_size))

        window_size = 10
        mse = np.Inf
        epoch = 0
        failed = 0
        old_dw = None
        while True:
            correct = 0
            for idx, row in enumerate(X):
                outputs = [row.tolist()]

                # FORWARD
                result = row
                first_array = result.reshape((1, -1))
                for i, new_array in enumerate(self.weights):
                    bias_row = new_array[-1]
                    new_array = new_array[:-1]
                    result = np.matmul(first_array, new_array)
                    result += bias_row
                    result = 1 / (1 + np.exp(result*-1))
                    outputs.append(result.tolist()[0])
                    first_array = result

                predict = np.argmax(result)

                # BACKWARD
                target = int(y[idx][0])
                # make one-hot from target
                target_oh = [0] * self.num_classes
                target_oh[target] = 1

                # update weights
                small_delta = []
                first = True
                for j, row_o in enumerate(reversed(outputs)):
                    if j == len(outputs)-1:
                        break
                    d_list = []
                    if first:
                        for k in range(len(row_o)):
                            d = (target_oh[k] - row_o[k]) * row_o[k] * (1 - row_o[k])
                            d_list.append(d)
                        small_delta.append(d_list)
                        first = False
                    else:
                        # calculate small deltas
                        for k in range(len(row_o)):
                            f_prime = row_o[k] * (1 - row_o[k])
                            rest = 0
                            for l in range(len(small_delta[0])):
                                rest += small_delta[0][l] * self.weights[-j][k][l]
                            f = (f_prime * rest)
                            d_list.append(f)
                        small_delta.insert(0, d_list)

                delta_w = []
                for mm in self.weights:
                    delta_w.append(np.zeros(mm.shape))

                for a, matrix in enumerate(reversed(self.weights)):
                    # calculate delta w
                    for b, row_m in enumerate(matrix):
                        # lr * small_delta(target) * output(source)
                        if b < len(matrix)-1:
                            for c, n in enumerate(row_m):
                                if old_dw is None:
                                    delta_w[-(a+1)][b][c] = self.lr * small_delta[-(a+1)][c] * outputs[-(a+2)][b]
                                else:
                                    delta_w[-(a + 1)][b][c] = (self.momentum * old_dw[-(a + 1)][b][c]) + \
                                                            (self.lr * small_delta[-(a + 1)][c] * outputs[-(a + 2)][b])
                        else:
                            for c, n in enumerate(row_m):
                                if old_dw is None:
                                    delta_w[-(a+1)][b][c] = self.lr * small_delta[-(a+1)][c] * 1.0
                                else:
                                    delta_w[-(a + 1)][b][c] = (self.momentum * old_dw[-(a + 1)][b][c]) + \
                                                              (self.lr * small_delta[-(a + 1)][c] * 1.0)

                # apply delta_w to weights
                for p in range(len(self.weights)):
                    self.weights[p] = np.add(self.weights[p], delta_w[p])
                old_dw = delta_w

                if predict == target:
                    correct += 1

            # STOP
            epoch += 1
            if self.epochs is None:
                # STOP CRITERIA
                new_mse = self.score(val_x, val_y, True)
                if new_mse < mse:
                    mse = new_mse
                    failed = 0
                else:
                    if failed == 0:
                        saved_weights = [np.copy(m) for m in self.weights]
                    failed += 1
                    if failed == window_size:
                        self.weights = [np.copy(m) for m in saved_weights]
                        break
            else:
                if epoch == self.epochs:
                    break

            # SHUFFLE
            if self.shuffle:
                X, y = self._shuffle_data(X, y)

        with open("out.txt", "a") as f:
            f.write(f"{self.momentum} {epoch}\n")

        self.epochs = epoch
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        predictions = []

        for row in X:

            result = row
            first_array = result.reshape((1, -1))
            for i, new_array in enumerate(self.weights):
                bias_row = new_array[-1]
                new_array = new_array[:-1]
                result = np.matmul(first_array, new_array)
                result += bias_row
                result = 1 / (1 + np.exp(result * -1))
                first_array = result

            predictions.append(result[0])

        return predictions

    def initialize_weights(self, input_length):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        weights = []
        last_layer = input_length

        if self.hidden_layer_widths is not None:
            for layer in self.hidden_layer_widths:
                if self.start_weights is None:
                    weights.append(np.random.normal(scale=self.lr, size=(last_layer + 1, layer)))
                else:
                    row = [self.start_weights] * ((last_layer + 1) * layer)
                    np_row = np.asarray(row)
                    np_row = np_row.reshape((last_layer + 1, layer))
                    weights.append(np_row)
                last_layer = layer
            if self.start_weights is None:
                weights.append(np.random.normal(scale=self.lr, size=(last_layer + 1, self.num_classes)))
            else:
                row = [self.start_weights] * ((last_layer + 1) * self.num_classes)
                np_row = np.asarray(row)
                np_row = np_row.reshape((last_layer+1, self.num_classes))
                weights.append(np_row)
        else:
            hidden_length = last_layer * 2
            if self.start_weights is None:
                weights.append(np.random.normal(scale=self.lr, size=(last_layer + 1, hidden_length)))
                weights.append(np.random.normal(scale=self.lr, size=(hidden_length + 1, self.num_classes)))
            else:
                row = [self.start_weights] * ((last_layer + 1) * hidden_length)
                np_row = np.asarray(row)
                np_row = np_row.reshape((last_layer + 1, hidden_length))
                weights.append(np_row)
                row = [self.start_weights] * ((hidden_length + 1) * self.num_classes)
                np_row = np.asarray(row)
                np_row = np_row.reshape((hidden_length + 1, self.num_classes))
                weights.append(np_row)

        return weights

    def score(self, X, y, MSE=False):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        predictions = self.predict(X)

        if MSE:
            expected = []
            for i in range(len(y)):
                new = [0] * self.num_classes
                new[int(y[i][0])] = 1
                expected.append(new)

            summation = 0
            for i in range(len(predictions)):
                for j in range(len(predictions[i])):
                    summation += (expected[i][j] - predictions[i][j]) ** 2

            return summation / len(predictions)
        else:
            am_predictions = [np.argmax(a) for a in predictions]

            correct = 0
            for i in range(len(predictions)):
                if am_predictions[i] == y[i][0]:
                    correct += 1

            return correct / len(predictions)

    def split_data(self, X, y, train_pct):
        # train_pct is the amount of train data desired [0, 1]
        X, y = self._shuffle_data(X, y)
        split_point = int(train_pct * len(X))
        train_X = X[:split_point]
        train_y = y[:split_point]
        test_X = X[split_point:]
        test_y = y[split_point:]
        return train_X, train_y, test_X, test_y

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        Xy = np.append(X, y, 1)
        np.random.shuffle(Xy)
        X = Xy[:, :-1]
        y = Xy[:, -1:]
        return X, y

    # ## Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights


if __name__ == '__main__':
    TEST = False
    DEBUG = False
    EVAL = False
    IRIS = False
    VOWEL = False
    SCIKIT_IRIS = False
    SCIKIT_VOWEL = False
    DIABETES = False
    GRID_CV = True
    RANDOM_CV = False

    if TEST:
        print("TEST")
        mat = Arff("Datasets/test.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        MLPClass = MLPClassifier(lr=1, momentum=0, shuffle=False, epochs=1, val_size=0,
                                 start_weights=1.0, hidden_layer_widths=[2])
        MLPClass.fit(data, labels)
        # print(MLPClass.weights)
        print("Train acc:", MLPClass.score(data, labels))

    if DEBUG:
        print("DEBUG")
        mat = Arff("Datasets/linsep2nonorigin.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        MLPClass = MLPClassifier(lr=0.1, momentum=0.5, shuffle=False, epochs=10, val_size=0, start_weights=0.0)
        MLPClass.fit(data, labels)
        print(MLPClass.weights)
        print("Train acc:", MLPClass.score(data, labels))

    if EVAL:
        print("EVAL")
        mat = Arff("Datasets/data_banknote_authentication.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        MLPClass = MLPClassifier(lr=0.1, momentum=0.5, shuffle=False, epochs=10, val_size=0, start_weights=0.0)
        MLPClass.fit(data, labels)
        print(MLPClass.weights)
        print("Train acc:", MLPClass.score(data, labels))

    if IRIS:
        print("IRIS")
        mat = Arff("Datasets/iris.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        MLPClass = MLPClassifier(lr=0.1, momentum=0.5, shuffle=True, epochs=None, val_size=0.2, start_weights=None)
        train_x, train_y, test_x, test_y = MLPClass.split_data(data, labels, 0.75)
        MLPClass.fit(train_x, train_y)
        # print(MLPClass.weights)
        print("Train acc:", MLPClass.score(train_x, train_y))
        print("Test acc:", MLPClass.score(test_x, test_y))
        print("Train MSE:", MLPClass.score(train_x, train_y, True))
        print("Test MSE:", MLPClass.score(test_x, test_y, True))
        print("Number of epochs:", MLPClass.epochs)

    if VOWEL:
        print("VOWEL")
        mat = Arff("Datasets/vowel.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        MLPClass = MLPClassifier(lr=0.1, momentum=0.5, shuffle=True, epochs=None, val_size=0.15, start_weights=None)
        train_x, train_y, test_x, test_y = MLPClass.split_data(data, labels, 0.75)
        MLPClass.fit(train_x, train_y)
        print("Train acc:", MLPClass.score(train_x, train_y))
        print("Test acc:", MLPClass.score(test_x, test_y))
        print("Train MSE:", MLPClass.score(train_x, train_y, True))
        print("Test MSE:", MLPClass.score(test_x, test_y, True))
        print("Number of epochs:", MLPClass.epochs)

    # scikit-learn MLP model
    if SCIKIT_IRIS:
        print()
        mat = Arff(arff="datasets/iris.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)

        PClass = MLPClassifier()
        train_X, train_y, test_X, test_y = PClass.split_data(data, labels, 0.75)
        train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)

        model = MLPC(verbose=1, learning_rate_init=0.1, solver='sgd', momentum=0.5, activation='logistic',
                     early_stopping=True, alpha=0.0001)  # , hidden_layer_sizes=(48,))
        model.fit(train_X, train_y)
        print("Training set accuracy:", round(model.score(train_X, train_y), 2))
        print("Test set accuracy:", round(model.score(test_X, test_y), 2))

    # scikit-learn MLP model
    if SCIKIT_VOWEL:
        print()
        mat = Arff(arff="datasets/vowel.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)

        PClass = MLPClassifier()
        train_X, train_y, test_X, test_y = PClass.split_data(data, labels, 0.75)
        train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)

        model = MLPC(learning_rate_init=0.1, solver='sgd', momentum=0.9, activation='logistic',
                     early_stopping=True, alpha=0.0001)  # , hidden_layer_sizes=(48,))
        model.fit(train_X, train_y)
        print("Training set accuracy:", round(model.score(train_X, train_y), 2))
        print("Test set accuracy:", round(model.score(test_X, test_y), 2))

    if DIABETES:
        print()
        mat = Arff(arff="datasets/diabetes.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)

        PClass = MLPClassifier()
        train_X, train_y, test_X, test_y = PClass.split_data(data, labels, 0.75)
        train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)

        model = MLPC(learning_rate_init=0.01, solver='sgd', momentum=0.2, activation='logistic',
                     early_stopping=True, alpha=0.0001)  # , hidden_layer_sizes=(48,))
        model.fit(train_X, train_y)
        print("Training set accuracy:", round(model.score(train_X, train_y), 2))
        print("Test set accuracy:", round(model.score(test_X, test_y), 2))

    if GRID_CV:
        mat = Arff(arff="datasets/vowel.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)

        PClass = MLPClassifier()
        train_X, train_y, test_X, test_y = PClass.split_data(data, labels, 0.75)
        train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)

        clf = GridSearchCV(MLPC(), {"learning_rate_init": [0.001, 0.01, 0.1, 0.5],
                                    "solver": ['sgd'],
                                    "momentum": [0.0, 0.2, 0.5, 0.9],
                                    "hidden_layer_sizes": [(20,), (40,), (60,), (80,), (100,)]})
        clf.fit(train_X, train_y)
        print(clf.best_params_)

    if RANDOM_CV:
        mat = Arff(arff="datasets/vowel.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)

        PClass = MLPClassifier()
        train_X, train_y, test_X, test_y = PClass.split_data(data, labels, 0.75)
        train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)

        rlf = RandomizedSearchCV(MLPC(), {"learning_rate_init": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
                                    "solver": ['sgd'],
                                    "momentum": [0.0, 0.2, 0.4, 0.5, 0.7, 0.9],
                                    "hidden_layer_sizes": [(20,), (40,), (60,), (80,), (100,)]})
        rlf.fit(train_X, train_y)
        print(rlf.best_params_)
