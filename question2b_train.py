import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

import question2b_utils

data = pd.read_csv(question2b_utils.TRAIN_FILE_PATH)

y = question2b_utils.get_target_values(data)
x = question2b_utils.get_data_for_model(data)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=4242)
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.2
params['max_depth'] = 4

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50, verbose_eval=10)

question2b_utils.store_model(bst)
