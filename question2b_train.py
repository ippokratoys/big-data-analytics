import pickle

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import question2b_utils

data = pd.read_csv(question2b_utils.TRAIN_FILE_PATH)

y = question2b_utils.get_target_values(data)
x = question2b_utils.get_data_for_model(data)

x_train, x_valid_test, y_train, y_valid_test = train_test_split(x, y, test_size=0.2, random_state=4242)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.5, random_state=4242)


def estimate(bst, x_test, y_test):
    x_matrix = xgb.DMatrix(x_test, label=y_test)
    prediction = bst.predict(x_matrix)
    pred_norm = [int(x > 0.5) for x in prediction]
    report = classification_report(y_test, pred_norm)
    metrics = question2b_utils.get_metrics(y_test, pred_norm)
    print("###### R E P O R T ######")
    print(report)
    print("###### M E T R I C S ######")
    print(metrics)
    return report


def train_bst(x_train, y_train, x_valid, y_valid, eta=0.2, max_depth=4, rounds=100):
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = eta
    params['max_depth'] = max_depth
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    bst = xgb.train(params, d_train, rounds, watchlist, early_stopping_rounds=50, verbose_eval=50)
    return bst


eta_list = [0.2]
depth_list = [4]
round_list = [10000]
reports = {}
for (eta, depth, rounds) in zip(eta_list, depth_list, round_list):
    print("Running for eta[{0}], depth[{1}], rounds[{2}]".format(eta, depth, rounds))
    bst = train_bst(x_train, y_train, x_valid, y_valid, eta, depth, rounds)
    report = estimate(bst, x_test, y_test)
    reports[(eta, depth, rounds)] = report
    question2b_utils.store_model(bst)

print(reports)
pickle.dump(reports, open("reports.pickle", "wb"))
