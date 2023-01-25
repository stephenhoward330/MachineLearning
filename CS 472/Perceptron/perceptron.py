import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron


class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, deterministic=None):
        """ Initialize class with chosen hyperparameters.
        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.weights = []
        self.epochs = 0

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.weights = self.initialize_weights(len(X[0])) if not initial_weights else initial_weights

        # add the column of ones for the bias
        X = np.append(X, np.ones([len(X), 1]), 1)

        epoch = 0
        accuracy = 0
        failed = 0
        while True:
            correct = 0
            for idx, row in enumerate(X):
                # find the total
                net = 0
                for j in range(len(row)):
                    net += row[j] * self.weights[j]

                # set to 0 or 1
                output = 1 if net > 0 else 0

                # compare it with target
                target = y[idx][0]
                target = int(target)
                if output != target:
                    delta_w = self.lr * (target - output) * row

                    # adjust weights
                    for k in range(len(self.weights)):
                        self.weights[k] += delta_w[k]
                else:
                    correct += 1

            epoch += 1
            # Misclassification Rate
            # print(1 - (correct / len(X)))
            if self.deterministic is None:
                new_accuracy = correct / len(X)
                if new_accuracy > accuracy:  # our accuracy has improved
                    accuracy = new_accuracy
                    failed = 0
                else:  # our accuracy has decreased
                    failed += 1
                    #  we stop when accuracy hasn't improved 4 epochs in a row
                    if failed == 4:
                        break
            else:
                if epoch == self.deterministic:
                    break
            if self.shuffle:
                X, y = self._shuffle_data(X, y)

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
        self.weights = self.initialize_weights(len(X[0])) if not self.weights else self.weights

        v = []
        for row in X:
            net = 0
            for j in range(len(row)):
                net += row[j] * self.weights[j]
            net += self.weights[-1]

            output = 1 if net > 0 else 0
            v.append(output)
        return v

    def predict_3(self, X):
        v = []
        for row in X:
            net = 0
            for j in range(len(row)):
                net += row[j] * self.weights[j]
            net += self.weights[-1]

            v.append(net)
        return v

    def initialize_weights(self, length):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        weights = [0] * (length + 1)
        return weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        prediction = self.predict(X)

        correct = 0
        for i in range(len(prediction)):
            if prediction[i] == y[i][0]:
                correct += 1

        return correct / len(prediction)

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

    # Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return [round(weight, 4) for weight in self.weights]

    def split_data(self, X, y, train_pct):
        # train_pct is the amount of train data desired [0, 1]
        X, y = self._shuffle_data(X, y)
        split_point = int(train_pct * len(X))
        train_X = X[:split_point]
        train_y = y[:split_point]
        test_X = X[split_point:]
        test_y = y[split_point:]
        return train_X, train_y, test_X, test_y

    def plot(self, X, y):
        # WILL ONLY WORK WITH DATA OF TWO INPUTS
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for i, row in enumerate(X):
            if y[i] == 1:
                x1.append(row[0])
                y1.append(row[1])
            else:
                x2.append(row[0])
                y2.append(row[1])
        plt.scatter(x1, y1, c='b', label="Class 1")
        plt.scatter(x2, y2, c='r', label="Class 2")

        slope = -1*(self.weights[0]/self.weights[1])
        y_int = -1*(self.weights[2]/self.weights[1])
        print(slope, y_int)
        x = np.linspace(-1, 1, 20)
        y = (slope * x) + y_int

        plt.plot(x, y, '-k', label="Decision Line")
        plt.title("Non-Linearly Separable Dataset")
        plt.legend(loc='lower left')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()


def run_model(file_name, lr=.1, shuffle=True, deterministic=None, label_count=1, split=None):
    file_name = "datasets/" + file_name
    mat = arff.Arff(arff=file_name, label_count=label_count)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)

    # Make Classifier
    PClass = PerceptronClassifier(lr=lr, shuffle=shuffle, deterministic=deterministic)
    if split is not None:
        train_X, train_y, test_X, test_y = PClass.split_data(data, labels, split)
        # Initial Misclassification Rate
        # print(PClass.score(train_X, train_y))
        PClass.fit(train_X, train_y)
        tr_accuracy = PClass.score(train_X, train_y)
        te_accuracy = PClass.score(test_X, test_y)
        print("Training Set Accuracy = [{:.2f}]".format(tr_accuracy))
        print("Test Set Accuracy = [{:.2f}]".format(te_accuracy))
    else:
        PClass.fit(data, labels)
        tr_accuracy = PClass.score(data, labels)
        print("Training Set Accuracy = [{:.2f}]".format(tr_accuracy))
    print("Final Weights =", PClass.get_weights())
    print("Learning Rate =", PClass.lr)
    print("Epochs =", PClass.epochs)
    # PClass.plot(data, labels)


def score_3(model1, model2, model3, X, y):
    prediction1 = model1.predict_3(X)
    prediction2 = model2.predict_3(X)
    prediction3 = model3.predict_3(X)

    prediction = []
    for i in range(len(prediction1)):
        if prediction1[i] > prediction2[i] and prediction1[i] > prediction3[i]:
            prediction.append(0)
        elif prediction2[i] > prediction1[i] and prediction2[i] > prediction3[i]:
            prediction.append(1)
        else:
            prediction.append(2)

    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == y[i]:
            correct += 1

    return correct / len(prediction)


def run_3_models(file_name, lr=.1, shuffle=True, deterministic=None, label_count=1):
    file_name = "datasets/" + file_name
    mat = arff.Arff(arff=file_name, label_count=label_count)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1)

    labels1 = [1 if label == 0 else 0 for label in labels]
    labels2 = [1 if label == 1 else 0 for label in labels]
    labels3 = [1 if label == 2 else 0 for label in labels]

    labels1 = np.array(labels1).reshape(-1, 1)
    labels2 = np.array(labels2).reshape(-1, 1)
    labels3 = np.array(labels3).reshape(-1, 1)

    # Make Classifier
    PClass1 = PerceptronClassifier(lr=lr, shuffle=shuffle, deterministic=deterministic)
    PClass2 = PerceptronClassifier(lr=lr, shuffle=shuffle, deterministic=deterministic)
    PClass3 = PerceptronClassifier(lr=lr, shuffle=shuffle, deterministic=deterministic)

    PClass1.fit(data, labels1)
    PClass2.fit(data, labels2)
    PClass3.fit(data, labels3)
    tr_accuracy = score_3(PClass1, PClass2, PClass3, data, labels)
    print("Training Set Accuracy = [{:.2f}]".format(tr_accuracy))
    print("Learning Rate =", lr)


if __name__ == "__main__":
    DEBUG = False
    EVAL = False
    LINSEP = False
    NOTLINSEP = False
    VOTING = False
    IRIS = True
    SCIKIT_VOTE = False
    SCIKIT_IRIS = False

    # debug data set
    if DEBUG:
        print("DEBUG DATA SET")
        run_model("linsep2nonorigin.arff", lr=0.1, shuffle=False, deterministic=10)
        print()

    # evaluation data set
    if EVAL:
        print("EVALUATION DATA SET")
        run_model("data_banknote_authentication.arff", lr=0.1, shuffle=False, deterministic=10)
        print()

    # lin sep data set
    if LINSEP:
        print("LINEARLY SEPARABLE DATA SET")
        run_model("mylinsep.arff", lr=0.1)
        print()

    # lin sep data set
    if NOTLINSEP:
        print("NOT-LINEARLY SEPARABLE DATA SET")
        run_model("mynotlinsep.arff", lr=0.1)
        print()

    # voting data set
    if VOTING:
        print("VOTING DATA SET")
        run_model("voting.arff", lr=0.1, split=0.7)
        print()

    # iris data set
    if IRIS:
        print("IRIS DATA SET")
        run_3_models("iris.arff", lr=0.1)
        print()

    # scikit-learn perceptron model with the voting data
    if SCIKIT_VOTE:
        mat = arff.Arff(arff="datasets/voting.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)

        PClass = PerceptronClassifier()
        train_X, train_y, test_X, test_y = PClass.split_data(data, labels, 0.7)
        train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)

        model = Perceptron(verbose=1)
        model.fit(train_X, train_y)
        print("Training set accuracy:", round(model.score(train_X, train_y), 2))
        print("Test set accuracy:", round(model.score(test_X, test_y), 2))

    # scikit-learn perceptron model with the iris data
    if SCIKIT_IRIS:
        mat = arff.Arff(arff="datasets/iris.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)

        PClass = PerceptronClassifier()
        train_X, train_y, test_X, test_y = PClass.split_data(data, labels, 0.7)
        train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)

        model = Perceptron(verbose=1)
        model.fit(train_X, train_y)
        print("Training set accuracy:", round(model.score(train_X, train_y), 2))
        print("Test set accuracy:", round(model.score(test_X, test_y), 2))
