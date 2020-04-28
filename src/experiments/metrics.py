import math
from sklearn.metrics import mean_squared_error, mean_squared_log_error


def rmse(y_true, y_pred):
    """
    Compute root mean squared error.

    :param y_true: list, ground truth (correct) target values.
    :param y_pred: list, estimated target values.
    """
    y_pred = [0.0 if x < 0 else x for x in y_pred]
    try:
        res = mean_squared_error(y_true, y_pred, squared=False)
    except ValueError as e:
        print(e)
        res = -1000
    return res


def nrmse(y_true, y_pred):
    """
    Compute normalized root mean squared error.

    :param y_true: list, ground truth (correct) target values.
    :param y_pred: list, estimated target values.
    """
    return rmse(y_true, y_pred) / (max(y_true) - min(y_true)) * 100


def rmsle(y_true, y_pred):
    """
    Compute root mean squared log error.

    :param y_true: list, ground truth (correct) target values.
    :param y_pred: list, estimated target values.
    """
    y_pred = [0.0 if x < 0 else x for x in y_pred]
    try:
        res = math.sqrt(mean_squared_log_error(y_true, y_pred))
    except ValueError as e:
        print(e)
        res = -1000
    return res
