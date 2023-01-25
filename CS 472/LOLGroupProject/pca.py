import pandas as pd
import numpy as np

data_directory = "data/"
pca_min_max = np.vsplit(np.loadtxt(data_directory + "min_max.csv", delimiter=","), [1])
pca_eigen_vectors = np.loadtxt(data_directory + "eigen_vectors.csv", delimiter=",")
pca_columns_order = ['assists_diff', 'CC_diff', 'visionScore_diff', 'towerKills_diff', 'dragonKills_diff', 'goldSpent', 'totalDamageDealtToChampions', 'deaths', 'baronKills', 'totalMinionsKilled', 'damage_per_death', 'damageDealtToObjectives', 'kills', 'totalHeal', 'win']
pca_mean = np.loadtxt(data_directory + "mean.csv", delimiter=",")


def convert_data(frame, num_eigen_vectors):
    """
    Convert any set of test data to the pca version. This includes normalizing the data. The min and max values of the
    data set are indicated in global variables that you should not modify. These values vere found based off all 1400
    rows that we currently have. The eigen vectors used to transform the data are also contained global variables that,
    once again, you shouldn't change.
    Args:
        frame: The data to convert as a pandas dataframe
        num_eigen_vectors: selects the num_eigen_vectors eigen vectors with the greatest eigen values

    Returns:
        matrix, same number of rows as data, number of columns is num_eigen_vectors
    """
    data = convert_dataframe(frame)
    data, null = normalize(data, pca_min_max)
    data -= pca_mean
    data = np.matmul(data, pca_eigen_vectors[:, 0:num_eigen_vectors])
    return data


def set_new_globals(frame):
    """
    If you have a different dataset you want to train over, use this function to recalculate the eigen vectors, min values and max values.
    Args:
        frame: The new set of data to calculate eigen vectors, min values and max values from. Should be a pandas data frame.

    Returns:
        array, the length of the array is equal to the number of new eigen vectors, each index contains what portion of
        the data is contained in each eigen vector
    """
    global pca_min_max, pca_eigen_vectors, pca_mean
    data = convert_dataframe(frame)
    data, pca_min_max = normalize_data(data, None, None)
    pca_mean = np.mean(data, axis=0)
    data = (data - pca_mean)
    covariance_matrix = np.cov(np.transpose(data))
    eigval, pca_eigen_vectors = np.linalg.eig(covariance_matrix)
    new_order = np.flip(np.argsort(eigval, axis=0))
    eigval = eigval[new_order]
    pca_eigen_vectors = pca_eigen_vectors[:, new_order]
    total_eigval = np.sum(eigval)
    percentage_eigval = eigval / total_eigval
    return percentage_eigval


def pca():
    data = pd.read_csv("combined.csv")
    set_new_globals(data)
    convert_data(data, 1)


def normalize_data(train_X, test_X, column_types):
    """
    A function to normalize data. You shouldn't have to use this
    Args:
        train_X: The data to normalize as a MxN matrix with each row being a different set of data and each column being each attribute
        test_X: Data that will be normalized, but will not factor into the min and max for normalization, can be None
        column_types: An array with a size equal to train_X.shape[1], indicates each type for the attributes. If the attribute is "continuous", it will be normalized, otherwise it won't

    Returns:
        train_X, test_X
            These are the data normalized. If test_X was None, then this function will return None for test_X
    """
    if column_types is None:
        train_X, min_max = normalize(train_X, None)
        if test_X is not None:
            test_X, min_max = normalize(test_X, min_max)
    else:
        for i in range(train_X.shape[1]):
            if column_types[i] == "continuous":
                train_X[:, [i]], min_max = normalize(np.resize(train_X[:, i], (train_X.shape[0], 1)), None)
                if test_X is not None:
                    test_X[:, [i]], min_max = normalize(np.resize(test_X[:, i], (test_X.shape[0], 1)), min_max)
    return train_X, test_X


def normalize(data, min_max_info):
    """
    Don't worry about this
    Args:
        data:
        min_max_info:

    Returns:

    """
    data = data
    if min_max_info is None:
        nan_indexes = np.argwhere(np.isnan(data))
        other_indexes = np.argwhere(data == data)
        default_value = data[other_indexes[0][0], other_indexes[0][1]]
        for index in nan_indexes:
            data[index[0], index[1]] = default_value
        max_values = np.resize(np.amax(data, axis=0), (1, data.shape[1]))
        min_values = np.resize(np.amin(data, axis=0), (1, data.shape[1]))
        for index in nan_indexes:
            data[index[0], index[1]] = float("nan")
    else:
        min_values = min_max_info[0]
        max_values = min_max_info[1]
    min_values_array = np.repeat(min_values, data.shape[0], axis=0)
    denominator = max_values - min_values
    denominator = np.repeat(denominator, data.shape[0], axis=0)
    data = (data - min_values_array) / denominator
    return data, (min_values, max_values)


def convert_dataframe(frame):
    """
    Don't worry about this.
    Args:
        frame:

    Returns:

    """
    data = frame.filter(pca_columns_order, axis=1)
    data = np.array(data)
    return data
