import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff
# from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, column_type=[], label_type='classification', weight_type='inverse_distance', k=3,
                 dist_type='euclidean'):
        """
        Args:
            column_type for each column tells you if continues[real] or if nominal[categorical].
            weight_type: inverse_distance voting or if non distance weighting.
                        Options = ["no_weight","inverse_distance"]
            label_type: Options = ["classification,"regression"]
        """
        self.column_type = column_type  # Note This won't be needed until part 5
        self.label_type = label_type
        self.weight_type = weight_type
        self.points = None
        self.labels = None
        self.k = k
        self.dist_type = dist_type

    def fit(self, data, labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            data (array-like): A 2D numpy array with the training data, excluding targets
            labels (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        self.points = data
        self.labels = labels

        return self

    def predict(self, data):
        """ Predict all classes for a data set X
        Args:
            data (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        results = []
        means = np.nanmean(self.points, axis=0)
        for new_row in data:  # tqdm
            if self.dist_type == 'euclidean':
                dists = np.linalg.norm(new_row - self.points, axis=1)
            elif self.dist_type == 'custom':
                dists = []
                for i, old_row in enumerate(self.points):
                    dist = 0
                    for j in range(len(old_row)):
                        a = new_row[j]
                        b = old_row[j]
                        if self.column_type[j] == 'continuous':
                            if np.isnan(a):
                                a = means[j]
                            if np.isnan(b):
                                b = means[j]
                            dist += (a-b) ** 2
                        elif self.column_type[j] == 'nominal':
                            if np.isnan(a) or np.isnan(b):
                                dist += 1
                            elif a == b:
                                dist += 0
                            else:
                                dist += 1
                        else:
                            raise Exception("bad column_type")
                    dists.append(np.sqrt(dist))
            else:
                raise Exception("invalid dist_type")

            min_dists = [np.inf] * self.k
            winners = [-1] * self.k
            for i, old_row in enumerate(self.points):
                max_pos = min_dists.index(max(min_dists))

                if dists[i] < min_dists[max_pos]:
                    min_dists[max_pos] = dists[i]
                    winners[max_pos] = self.labels[i]

            for j in range(len(min_dists)):
                if min_dists[j] == 0:
                    min_dists[j] = 0.00000000001

            if self.label_type == "classification":

                if self.weight_type == "inverse_distance":
                    local_label = []
                    local_nums = []
                    for i in range(len(winners)):
                        if winners[i] in local_label:
                            idx = local_label.index(winners[i])
                            local_nums[idx] += 1 / (min_dists[i] ** 2)
                        else:
                            local_label.append(winners[i])
                            local_nums.append(1 / (min_dists[i] ** 2))
                    max_pos = local_nums.index(max(local_nums))
                    winner = local_label[max_pos]
                elif self.weight_type == "no_weight":
                    winner = max(set(winners), key=winners.count)
                else:
                    raise Exception("invalid weight_type")
            elif self.label_type == "regression":
                if self.weight_type == "inverse_distance":
                    numerator, denominator = 0, 0
                    for i in range(len(winners)):
                        numerator += winners[i] / (min_dists[i] ** 2)
                        denominator += 1 / (min_dists[i] ** 2)
                    winner = numerator / denominator
                elif self.weight_type == "no_weight":
                    winner = np.mean(winners)
                else:
                    raise Exception("invalid weight_type")
            else:
                raise Exception("invalid label_type")

            results.append(winner)

        return results

    # Returns the Mean score given input data and labels
    def score(self, X, y):
        """ Return accuracy of model on a given data set (MSE if regression). Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """
        predictions = self.predict(X)

        if self.label_type == "classification":
            # ACCURACY
            correct = 0
            for i in range(len(predictions)):
                if predictions[i] == y[i]:
                    correct += 1

            return round(100 * (correct / len(y)), 2)
        elif self.label_type == "regression":
            # MSE
            se = 0
            for i in range(len(predictions)):
                se += (predictions[i] - y[i]) ** 2

            return round(se / len(y), 6)
        else:
            raise Exception("invalid label_type")


def normalize(data, labels, column_types):
    maxes = np.nanmax(data, axis=0)
    mins = np.nanmin(data, axis=0)
    for j in range(len(data[0])):
        if column_types[j] == 'continuous':
            for i in range(len(data)):
                if not np.isnan(data[i][j]):
                    data[i][j] = (data[i][j] - mins[j]) / (maxes[j] - mins[j])

    if column_types[-1] == 'continuous':
        max_l = max(labels)
        min_l = min(labels)
        for i in range(len(labels)):
            if not np.isnan(labels[i]):
                labels[i] = (labels[i] - min_l) / (max_l - min_l)

    return data, labels


if __name__ == '__main__':
    DEBUG = False
    EVAL = False
    TELESCOPE = False
    HOUSING = False
    CREDIT = False
    SK_TELESCOPE = False
    SK_HOUSING = True

    if DEBUG:
        print("DEBUG")
        mat = Arff("datasets/seismic-bumps_train.arff", label_count=1)
        mat2 = Arff("datasets/seismic-bumps_test.arff", label_count=1)
        raw_data = mat.data
        train_data = raw_data[:, :-1]
        train_labels = raw_data[:, -1]

        raw_data2 = mat2.data
        test_data = raw_data2[:, :-1]
        test_labels = raw_data2[:, -1]

        KNN = KNNClassifier()
        KNN.fit(train_data, train_labels)
        # pred = KNN.predict(test_data)
        score = KNN.score(test_data, test_labels)
        print("Acc:", score)
        # np.savetxt("my-seismic-bump-prediction.csv", pred, delimiter=',', fmt="%i")
        print()

    if EVAL:
        print("EVAL")
        mat = Arff("datasets/diabetes.arff", label_count=1)
        mat2 = Arff("datasets/diabetes_test.arff", label_count=1)
        raw_data = mat.data
        train_data = raw_data[:, :-1]
        train_labels = raw_data[:, -1]

        raw_data2 = mat2.data
        test_data = raw_data2[:, :-1]
        test_labels = raw_data2[:, -1]

        KNN = KNNClassifier()
        KNN.fit(train_data, train_labels)
        # pred = KNN.predict(test_data)
        score = KNN.score(test_data, test_labels)
        print("Acc:", score)
        # np.savetxt("diabetes-prediction.csv", pred, delimiter=',', fmt="%i")
        print()

    if TELESCOPE:
        print("TELESCOPE")
        mat = Arff("datasets/telescope_train.arff", label_count=1)
        mat2 = Arff("datasets/telescope_test.arff", label_count=1)
        raw_data = mat.data
        train_data = raw_data[:, :-1]
        train_labels = raw_data[:, -1]

        raw_data2 = mat2.data
        test_data = raw_data2[:, :-1]
        test_labels = raw_data2[:, -1]

        # NORMALIZE
        train_data, train_labels = normalize(train_data, train_labels, mat.attr_types)
        test_data, test_labels = normalize(test_data, test_labels, mat.attr_types)

        KNN = KNNClassifier(weight_type="inverse_distance", k=3)
        KNN.fit(train_data, train_labels)
        score = KNN.score(test_data, test_labels)
        print("Acc:", score)
        print()

    if HOUSING:
        print("HOUSING")
        mat = Arff("datasets/housing_train.arff", label_count=1)
        mat2 = Arff("datasets/housing_test.arff", label_count=1)
        raw_data = mat.data
        train_data = raw_data[:, :-1]
        train_labels = raw_data[:, -1]

        raw_data2 = mat2.data
        test_data = raw_data2[:, :-1]
        test_labels = raw_data2[:, -1]

        # NORMALIZE
        train_data, train_labels = normalize(train_data, train_labels, mat.attr_types)
        test_data, test_labels = normalize(test_data, test_labels, mat.attr_types)

        KNN = KNNClassifier(weight_type="inverse_distance", label_type="regression", k=3)
        KNN.fit(train_data, train_labels)
        score = KNN.score(test_data, test_labels)
        print("MSE:", score)
        print()

    if CREDIT:
        print("CREDIT")
        mat = Arff("datasets/credit.arff", label_count=1)
        raw_data = mat.data
        train_data = raw_data[90:, :-1]
        train_labels = raw_data[90:, -1]

        test_data = raw_data[:90, :-1]
        test_labels = raw_data[:90, -1]

        # NORMALIZE
        train_data, train_labels = normalize(train_data, train_labels, mat.attr_types)
        test_data, test_labels = normalize(test_data, test_labels, mat.attr_types)

        KNN = KNNClassifier(dist_type="custom", k=3, column_type=mat.attr_types)
        KNN.fit(train_data, train_labels)
        score = KNN.score(test_data, test_labels)
        print("Acc:", score)
        print()

    if SK_TELESCOPE:
        print("SK TELESCOPE")
        mat = Arff("datasets/telescope_train.arff", label_count=1)
        mat2 = Arff("datasets/telescope_test.arff", label_count=1)
        raw_data = mat.data
        train_data = raw_data[:, :-1]
        train_labels = raw_data[:, -1]

        raw_data2 = mat2.data
        test_data = raw_data2[:, :-1]
        test_labels = raw_data2[:, -1]

        # NORMALIZE
        train_data, train_labels = normalize(train_data, train_labels, mat.attr_types)
        test_data, test_labels = normalize(test_data, test_labels, mat.attr_types)

        KNN = KNeighborsClassifier(n_neighbors=3, p=1, weights='distance')
        KNN.fit(train_data, train_labels)
        score = KNN.score(test_data, test_labels)
        print("Acc:", score)
        print()

    if SK_HOUSING:
        print("SK HOUSING")
        mat = Arff("datasets/housing_train.arff", label_count=1)
        mat2 = Arff("datasets/housing_test.arff", label_count=1)
        raw_data = mat.data
        train_data = raw_data[:, :-1]
        train_labels = raw_data[:, -1]

        raw_data2 = mat2.data
        test_data = raw_data2[:, :-1]
        test_labels = raw_data2[:, -1]

        # NORMALIZE
        train_data, train_labels = normalize(train_data, train_labels, mat.attr_types)
        test_data, test_labels = normalize(test_data, test_labels, mat.attr_types)

        KNN = KNeighborsRegressor(n_neighbors=3, weights='uniform', p=1)
        KNN.fit(train_data, train_labels)
        score = KNN.score(test_data, test_labels)
        print("R^2:", score)
        print()
