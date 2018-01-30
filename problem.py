import os
import pandas as pd
import rampwf as rw
import numpy as np
from rampwf.score_types.base import BaseScoreType

problem_title = 'Bike share in Boston'
_target_column_names = [
    'nb of bikes'
]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=_target_column_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()


# Useful for a score (see below)

docks_by_station = {0: 15, 1: 15, 2: 15, 3: 15, 4: 15, 5: 19, 6: 11, 7: 15, 8: 15, 9: 19, 10: 21, 11: 15, 12: 19,
 13: 15, 14: 15, 15: 19, 16: 25, 17: 46, 18: 21, 19: 19, 20: 15, 21: 15, 22: 15, 23: 15, 24: 15, 25: 15, 26: 11,
 27: 15, 28: 23, 29: 25, 30: 15, 31: 19, 32: 19, 33: 22, 34: 15, 35: 19, 36: 19, 37: 19, 38: 18, 39: 19, 40: 19,
 41: 15, 42: 15, 43: 18, 44: 19, 45: 15, 46: 15, 47: 15, 48: 15, 49: 15, 50: 15, 51: 19, 52: 15, 53: 19, 54: 15,
 55: 27, 56: 19, 57: 19, 58: 23, 59: 15, 60: 15, 61: 15, 62: 19, 63: 15, 64: 17, 65: 15, 66: 19, 67: 35, 68: 15,
 69: 15, 70: 15, 71: 19, 72: 15, 73: 19, 74: 19, 75: 19, 76: 19, 77: 15, 78: 11, 79: 15, 80: 19, 81: 19, 82: 19,
 83: 15, 84: 25, 85: 15, 86: 15, 87: 19, 88: 19, 89: 19, 90: 17, 91: 34, 92: 15, 93: 15, 94: 15, 95: 15, 96: 19,
 97: 19, 98: 19, 99: 19, 100: 15, 101: 17, 102: 15, 103: 15, 104: 15, 105: 15, 106: 15, 107: 19, 108: 15, 109: 15,
 110: 17, 111: 19, 112: 15, 113: 15, 114: 19, 115: 17, 116: 15, 117: 23, 118: 23, 119: 19, 120: 19, 121: 18,
 122: 19, 123: 19, 124: 15, 125: 15, 126: 14, 127: 18, 128: 15, 129: 18, 130: 15, 131: 14, 132: 18, 133: 14, 134: 15,
 135: 16, 136: 15, 137: 19, 138: 19, 139: 25, 140: 19, 141: 19, 142: 23, 143: 19, 144: 15, 145: 14, 146: 23, 147: 37,
 148: 19, 149: 15, 150: 15, 151: 23, 152: 11, 153: 15, 154: 15, 155: 15, 156: 15, 157: 15, 158: 15, 159: 15, 160: 15,
 161: 15, 162: 19, 163: 33, 164: 16, 165: 15, 166: 15, 167: 19, 168: 19, 169: 15, 170: 15}



# Mean Absolute Error

class MAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mae', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred)))

# New Error
# To penalize more if one did not predict a 'risk zone'

class Danger_Score(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='danger', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = 0
        n = len(y_true)
        for i in range(np.int(n / 171)):
            idx = 171 * i
            for j in range(171):
                idx_tmp = idx + j
                tmp = abs(y_true[idx_tmp] - y_pred[idx_tmp])
                score += tmp

                # if the true value is "at risk", but not the predicted value
                if y_true[idx_tmp] >= np.int(docks_by_station[j] * 0.9):
                    if y_pred[idx_tmp] < np.int(docks_by_station[j] * 0.9):
                        score += 2 * tmp
                elif y_true[idx_tmp] <= np.int(docks_by_station[j] * 0.1) + 1:
                    if y_pred[idx_tmp] > np.int(docks_by_station[j] * 0.1):
                        score += 2 * tmp

                # if the true value is not "at risk", but the predicted value is
                else:
                    if y_pred[idx_tmp] >= np.int(docks_by_station[j] * 0.9):
                        score += tmp
                    if y_pred[idx_tmp] <= np.int(docks_by_station[j] * 0.1) + 1:
                        score += tmp
        return (score / n)



score_types = [
    Danger_Score(name='danger', precision=2),
    MAE(name='mae', precision=2),
]


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_names].values
    X_df = data.drop(_target_column_names, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        # 3 first months (around 60 days)
        return X_df[:984960], y_array[:984960]
    else:
        return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv.bz2'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv.bz2'
    return _read_data(path, f_name)

NB_STATIONS = 171
NB_SLOTS = 96
DAY_CV = 7


# Classical way to do a CV when dealing with Time series
# X is not needed actually, only here for syntax / compatibility reasons

def get_cv(X, y):
    n_folds = 5
    len_total = len(y)
    cst = NB_STATIONS * NB_SLOTS * DAY_CV
    train_begin = 0
    for i in range(n_folds):
        train_end = len_total - 1 - (n_folds - i) * cst
        test_begin = train_end + 1
        test_end = test_begin + cst
        train_index = np.arange(train_begin, train_end)
        test_index = np.arange(test_begin, test_end)
        yield train_index, test_index
