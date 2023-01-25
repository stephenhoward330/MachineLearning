import numpy as np
from arff import Arff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydot

# ## NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score


class Node:
    def __init__(self, counts, used, parent_most_common):
        self.children = []
        self.label = None
        self.split_on = None
        self.parent_most_common = parent_most_common
        self.counts = counts
        self.used = used

    def get_prediction(self, instance):
        if self.label is not None:
            return self.label
        elif self.split_on is not None:
            return self.children[int(instance[int(self.split_on)])].get_prediction(instance)
        else:
            return self.parent_most_common

    def fill_tree(self, X, y):
        # if pure, return node
        if len(np.unique(y)) == 1:
            self.label = y[0][0]
            return self
        # if empty, return most common class of parent
        elif len(X) == 0:
            self.label = self.parent_most_common
            return self
        # if no more attributes, return most common class of parent
        elif False not in self.used:
            self.label = self.parent_most_common
            return self

        # calc gain for each attribute
        info = 0
        _, u_counts = np.unique(y, return_counts=True)
        for n in u_counts:
            if n != 0:
                info -= (n / len(y)) * np.log2(n / len(y))
        # print("info: ", info)

        combined = np.append(X, y, 1)
        gains = []
        # get gain for each attribute
        for i in range(len(self.used)):
            if self.used[i]:
                gains += [None]
            else:
                # calc gain
                loss = 0
                x_values, x_occurrences = np.unique(X[:, i], return_counts=True)
                y_values, y_occurrences = np.unique(y, return_counts=True)
                for j in range(len(x_values)):
                    log_num = 0
                    for k in range(len(y_values)):
                        c = 0
                        for row in combined:
                            if row[i] == x_values[j] and row[len(row)-1] == y_values[k]:
                                c += 1
                        if c != 0:
                            log_num -= ((c / x_occurrences[j]) * np.log2(c / x_occurrences[j]))
                    loss += (x_occurrences[j] / len(X)) * log_num
                gains += [info-loss]
        # print("gains: ", gains)

        # select highest gain
        highest_i = 0
        maximum = -1
        for i in range(len(gains)):
            if gains[i] is None:
                continue
            elif gains[i] > maximum:
                highest_i = i
                maximum = gains[i]
        if maximum == -1:
            raise Exception("ALL NONE")

        # create new node for each partition
        self.used[highest_i] = True
        self.split_on = highest_i
        for i in range(self.counts[highest_i]):
            # reduce x and y
            new_combined = [row for row in combined if row[highest_i] == i]
            new_combined = np.asarray(new_combined)
            if new_combined.size == 0:
                new_x = []
                new_y = []
            else:
                new_x = new_combined[:, :-1]
                new_y = new_combined[:, -1:]

            # get the most common value
            v, c = np.unique(y, return_counts=True)
            ind = np.argmax(c)

            # recur
            new_node = Node(self.counts, self.used.copy(), v[ind])
            new_node.fill_tree(new_x, new_y)
            self.children.append(new_node)
        return self


class DTClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, counts=None):
        """ Initialize class with chosen hyper-parameters.
        Args:
        Optional Args (Args we think will make your life easier):
            counts: A list of Ints that tell you how many types of each feature there are
        Example:
            DT  = DTClassifier()
            or
            DT = DTClassifier(count = [2,3,2,2])
            Data_set =
            [[0,1,0,0],
            [1,2,1,1],
            [0,1,1,0],
            [1,2,0,1],
            [0,0,1,1]]
        """
        self.counts = counts
        self.root = None

    def fit(self, X, y):
        """ Fit the data; Make the Decision tree
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.root = Node(self.counts, [False] * (len(self.counts)-1), None)
        self.root = self.root.fill_tree(X, y)

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        results = []
        for instance in X:
            results.append(self.root.get_prediction(instance))
        return results

    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given data set. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets
        """
        predictions = self.predict(X)

        correct = 0
        for j in range(len(predictions)):
            if predictions[j] == y[j]:
                correct += 1
        return round(correct / len(predictions), 2)

    def get_test_data(self, X, y, start, stop):
        X, y = self._shuffle_data(X, y)
        train_X = np.vstack((X[:start], X[stop:]))
        train_y = np.vstack((y[:start], y[stop:]))
        test_X = X[start:stop]
        test_y = y[start:stop]
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


if __name__ == '__main__':
    PIZZA = False
    DEBUG = False
    EVAL = False
    CARS = False
    VOTING = False
    SKL_CARS = False
    SKL_VOTING = False
    SOYBEAN = True

    if PIZZA:
        mat = Arff("datasets/pizza.arff", label_count=1)

        counts = []  # # this is so you know how many types for each column
        for i in range(mat.data.shape[1]):
            counts += [mat.unique_value_count(i)]

        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        DTClass = DTClassifier(counts)
        DTClass.fit(data, labels)

        # pred = DTClass.predict(data)
        Acc = DTClass.score(data, labels)
        # print(pred)
        print("Accuracy = [{:.2f}]".format(Acc))

    if DEBUG:
        mat = Arff("datasets/lenses.arff", label_count=1)

        counts = []  # # this is so you know how many types for each column
        for i in range(mat.data.shape[1]):
            counts += [mat.unique_value_count(i)]

        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        DTClass = DTClassifier(counts)
        DTClass.fit(data, labels)

        # pred = DTClass.predict(data)
        Acc = DTClass.score(data, labels)
        # print(pred)
        print("Train accuracy = [{:.2f}]".format(Acc))

        mat2 = Arff("datasets/all_lenses.arff", label_count=1)
        data2 = mat2.data[:, 0:-1]
        labels2 = mat2.data[:, -1]

        # pred2 = DTClass.predict(data2)
        Acc2 = DTClass.score(data2, labels2)
        # print(pred2)
        print("Test accuracy = [{:.2f}]".format(Acc2))

    if EVAL:
        mat = Arff("datasets/zoo.arff", label_count=1)

        counts = []  # # this is so you know how many types for each column
        for i in range(mat.data.shape[1]):
            counts += [mat.unique_value_count(i)]

        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        DTClass = DTClassifier(counts)
        DTClass.fit(data, labels)

        Acc = DTClass.score(data, labels)
        print("Train accuracy = [{:.2f}]".format(Acc))

        mat2 = Arff("datasets/all_zoo.arff", label_count=1)
        data2 = mat2.data[:, 0:-1]
        labels2 = mat2.data[:, -1]

        # pred = DTClass.predict(data2)
        # np.savetxt("pred_zoo.csv", pred, delimiter=",")
        Acc2 = DTClass.score(data2, labels2)
        print("Test accuracy = [{:.2f}]".format(Acc2))

    if CARS:
        print("CARS")
        mat = Arff("datasets/cars.arff", label_count=1)

        counts = []  # # this is so you know how many types for each column
        for i in range(mat.data.shape[1]):
            counts += [mat.unique_value_count(i)]
        # print(len(counts))

        data = mat.data[:, 0:-1]
        # print(len(data))
        labels = mat.data[:, -1].reshape(-1, 1)
        DTClass = DTClassifier(counts)

        start = 0
        for i in range(10):
            stop = int(((i + 1) * 0.1) * len(data))
            # print(start, stop)
            train_x, train_y, test_x, test_y = DTClass.get_test_data(data, labels, start, stop)
            DTClass.fit(train_x, train_y)

            Acc = DTClass.score(train_x, train_y)
            Acc2 = DTClass.score(test_x, test_y)
            print(f"Fold {i+1}:  |  Train accuracy: {Acc}  |  Test accuracy: {Acc2}")

            start = stop

    if VOTING:
        print("VOTING")
        mat = Arff("datasets/voting.arff", label_count=1)

        counts = []  # # this is so you know how many types for each column
        for i in range(mat.data.shape[1]):
            counts += [mat.unique_value_count(i)]
        # print(len(counts))

        data = mat.data[:, 0:-1]
        # print(len(data))
        labels = mat.data[:, -1].reshape(-1, 1)

        # handle missing values
        for idx, column, in enumerate(data.T):
            inc_count = False
            for jdx, num in enumerate(column):
                if np.isnan(num):
                    data[jdx][idx] = counts[idx]
                    inc_count = True
            if inc_count:
                counts[idx] += 1

        # print(counts)
        # print(data)
        DTClass = DTClassifier(counts)

        start = 0
        for i in range(10):
            stop = int(((i + 1) * 0.1) * len(data))
            # print(start, stop)
            train_x, train_y, test_x, test_y = DTClass.get_test_data(data, labels, start, stop)
            DTClass.fit(train_x, train_y)

            Acc = DTClass.score(train_x, train_y)
            Acc2 = DTClass.score(test_x, test_y)
            print(f"Fold {i + 1}:  |  Train accuracy: {Acc}  |  Test accuracy: {Acc2}")

            start = stop

    if SKL_CARS:
        print("SKL_CARS")
        mat = Arff("datasets/cars.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        my_DTClass = DTClassifier([])
        skl_DTClass = DecisionTreeClassifier()
        # DTClass.fit(data, labels)
        # print(DTClass.score(data, labels))

        start = 0
        for i in range(10):
            stop = int(((i + 1) * 0.1) * len(data))
            # print(start, stop)
            train_x, train_y, test_x, test_y = my_DTClass.get_test_data(data, labels, start, stop)
            skl_DTClass.fit(train_x, train_y)

            Acc = skl_DTClass.score(train_x, train_y)
            Acc2 = skl_DTClass.score(test_x, test_y)
            print(f"Fold {i + 1}:  |  Train accuracy: {Acc}  |  Test accuracy: {Acc2}")

            start = stop

    if SKL_VOTING:
        print("SKL_VOTING")
        mat = Arff("datasets/voting.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        my_DTClass = DTClassifier([])
        skl_DTClass = DecisionTreeClassifier()
        # DTClass.fit(data, labels)
        # print(DTClass.score(data, labels))

        counts = []  # # this is so you know how many types for each column
        for i in range(mat.data.shape[1]):
            counts += [mat.unique_value_count(i)]

        # handle missing values
        for idx, column, in enumerate(data.T):
            inc_count = False
            for jdx, num in enumerate(column):
                if np.isnan(num):
                    data[jdx][idx] = counts[idx]

        start = 0
        for i in range(10):
            stop = int(((i + 1) * 0.1) * len(data))
            # print(start, stop)
            train_x, train_y, test_x, test_y = my_DTClass.get_test_data(data, labels, start, stop)
            skl_DTClass.fit(train_x, train_y)

            Acc = skl_DTClass.score(train_x, train_y)
            Acc2 = skl_DTClass.score(test_x, test_y)
            print(f"Fold {i + 1}:  |  Train accuracy: {Acc}  |  Test accuracy: {Acc2}")

            start = stop

    if SOYBEAN:
        print("SOYBEAN")
        mat = Arff("datasets/soybean.arff", label_count=1)
        data = mat.data[:, 0:-1]
        labels = mat.data[:, -1].reshape(-1, 1)
        my_DTClass = DTClassifier([])
        skl_DTClass = DecisionTreeClassifier(criterion='entropy')
        # DTClass.fit(data, labels)
        # print(DTClass.score(data, labels))

        counts = []  # # this is so you know how many types for each column
        for i in range(mat.data.shape[1]):
            counts += [mat.unique_value_count(i)]
        print(len(counts))

        # handle missing values
        for idx, column, in enumerate(data.T):
            inc_count = False
            for jdx, num in enumerate(column):
                if np.isnan(num):
                    data[jdx][idx] = counts[idx]

        start = 0
        test_acc = []
        for i in range(10):
            stop = int(((i + 1) * 0.1) * len(data))
            # print(start, stop)
            train_x, train_y, test_x, test_y = my_DTClass.get_test_data(data, labels, start, stop)
            skl_DTClass.fit(train_x, train_y)

            if i == 0:
                tree.export_graphviz(skl_DTClass, out_file='tree.dot', max_depth=2)

            Acc = skl_DTClass.score(train_x, train_y)
            Acc2 = skl_DTClass.score(test_x, test_y)
            test_acc.append(Acc2)
            print(f"Fold {i + 1}:  |  Train accuracy: {Acc}  |  Test accuracy: {Acc2}")

            start = stop
        print("Average test accuracy:", np.mean(test_acc))
