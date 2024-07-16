import os

import pandas as pd
import xgboost as xgb

from util import is_essential_agreement, essential_agreement_cus_metric

# Configuration constants
TRAIN_DATA = '../genome-data-ignore/svm/train-10mer-nt.svm'
TEST_DATA = '../genome-data-ignore/svm/test-10mer-nt.svm'
RESULT_DIR = 'result/'

# Load the dataset in DMatrix format
dtrain = xgb.DMatrix(f'{TRAIN_DATA}?format=libsvm')
dtest = xgb.DMatrix(f'{TEST_DATA}?format=libsvm')

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.1,
    'nthread': 76,
    'tree_method': 'approx',  # Using histogram-based method
}

# Watchlist to observe the training and testing performance
watchlist = [(dtrain, 'train'), (dtest, 'test')]

# Train the model with the custom metric
num_boost_round = 300
bst = xgb.train(params, dtrain, num_boost_round, watchlist, custom_metric=essential_agreement_cus_metric)

# Print the final evaluation results
evals_result = bst.eval(dtest)
print(evals_result)

# Predict using the trained model
predictions = bst.predict(dtest)

# Create a dataframe with predictions and actual labels
results_df = pd.DataFrame({
    'Prediction': predictions,
    'Actual': dtest.get_label(),
    'Essential Agreement': list(is_essential_agreement(dtest.get_label(), predictions))
})

# Save the dataframe to a csv file
results_df.to_csv(os.path.join(RESULT_DIR, "predictions.csv"), index=False)
