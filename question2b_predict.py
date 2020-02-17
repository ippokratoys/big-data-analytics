import pandas as pd
import xgboost as xgb

import question2b_utils

data = pd.read_csv(question2b_utils.TEST_FILE_PATH)

ids = data.Id
x = question2b_utils.get_data_for_model(data)

print("Loading model...")
bst = question2b_utils.load_model()

x_matrix = xgb.DMatrix(x, label=ids)

print("Predicting data...")
prediction = bst.predict(x_matrix)

print("Preparing data for csv...")
data_to_submit = {"Id": [], "Predicted": []}
for (index, elem) in enumerate(prediction):
    data_to_submit["Id"].append(ids[index])
    data_to_submit["Predicted"].append(int(elem > 0.5))

print("Preparing data for csv...")
question2b_utils.dump_predictions_to_csv(data_to_submit)
