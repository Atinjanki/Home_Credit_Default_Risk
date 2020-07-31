
from lime.lime_tabular import LimeTabularExplainer
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import random
from joblib import dump
from joblib import load
from sklearn.preprocessing import MinMaxScaler, Imputer
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")


df_train = pd.read_csv(r'data\application_train.csv')
df_test = pd.read_csv(r'data\application_test.csv')

sample_submission = pd.read_csv(r'data\sample_submission.csv')

try:
    assert len(df_train.dropna()) == len(
        df_train) and len(df_test.dropna()) == len(df_test)
except AssertionError:
    print('Handling NA Values that were found!')
    # ### Set Missing values as np.nan
    # Also np.nan is understood as missing values by XGBOOST by default
    df_train.fillna(np.nan, inplace=True)
    df_test.fillna(np.nan, inplace=True)


# ### Encode Categorical values to discrete values


# one-hot encoding
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

#print('df_train shape: ', df_train.shape)
#print('df_test shape: ', df_test.shape)

# *Assert inequality here
try:
    assert df_train.shape[1] == (df_test.shape[1]+1)
except AssertionError:
    print('Handling train and test data size mismatch!')

    # keeping aside TARGET label;s for a moment
    train_labels_impute = df_train['TARGET']

    # Align df_train wrt df_test, such that only common columns are present in both
    df_train, df_test = df_train.align(df_test, join='inner', axis=1)

    # Add TARGET column back to df_train
    df_train['TARGET'] = train_labels_impute


assert df_train.shape[1] == (df_test.shape[1]+1)
#print('df_train shape: ', df_train.shape)
#print('df_test shape: ', df_test.shape)


# ### Update Missing Values and make a copy of Test and Train data

# store a copy not normal assignment
train_impute = df_train.drop(columns=['TARGET'])

# Feature names
features_impute = list(train_impute.columns)

# use median for missing values
imputer = Imputer(strategy='median')

# Use MinMax Scalar to bring the features to the range of (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))

# Lets fit training data
imputer.fit(train_impute)

# Transforming df_train and df_test data
# remember we have dropped TARGET column
train_impute = imputer.transform(train_impute)
test_impute = imputer.transform(df_test)

# Similar transformation with MInMax scaler
scaler.fit(train_impute)
train_impute = scaler.transform(train_impute)
test_impute = scaler.transform(test_impute)

assert train_impute.shape[1] == test_impute.shape[1]
#print('train shape: ', train_impute.shape)
#print('test shape: ', test_impute.shape)


# ## Lets Train Models
"""
We trained Logistic Regression, Random Forest, SVM SVC and XGBOOST. All after handling missing values.
We also trained XGBOOST with missing values.
We will discuss only XGBOOST (with imputation) in this code.
Accuracy values for rest of the models are:
    LOG_REG - 0.68233
    RF      - 0.69466
    SVM-SVC - 
    XGBOOST - 0.74427
    XGBOOST - 0.74410 (with missing values)
"""


# ### Train XGBoost

seed = 12345

params = {'objective': 'binary:logistic', 'learning_rate': 0.1,
          'gamma': 1.0, 'min_child_weight': 0.1,
          'max_depth': 6, 'n_estimators': 200, 'random_state': seed}


#xgbModel = XGBClassifier(**params)

# Train on the training data
#xgbModel.fit(train_impute, train_labels_impute)

# Save the model to file
#dump(xgbModel, "xgb_imputation.joblib.dat")
#print("Saved model to: xgb_imputation.joblib.dat")


# load model from file
xgbModel = load("xgb_imputation.joblib.dat")
print("Loaded model from: xgb_imputation.joblib.dat")


# Extract feature importances
feature_importance_values_xgbModel = xgbModel.feature_importances_
feature_importances_xgbModel = pd.DataFrame(
    {'feature': features_impute, 'importance': feature_importance_values_xgbModel})

# Making predictions on test data
predictions_xgbModel = xgbModel.predict_proba(test_impute)[:, 1]


# For Kaggle submission
submission = df_test[['SK_ID_CURR']]
submission['TARGET'] = predictions_xgbModel

# Check the submission file is correct or not, as per Kaggle requirements
assert len(sample_submission) == len(submission)

# Save the submission dataframe
submission.to_csv('XGBClassifier_with_imputation.csv', index=False)

# Print top-10 features
print(feature_importances_xgbModel.head(15).to_dict('records'))

# Feature Importance plot
# feature_importances_xgbModel.plot(kind='bar',x='feature',y='importance')

"""
### As CODE_GENDER_M (5th) and CODE_GENDER_F (11th) are in top-15 features of the model.
### We should make a check for fairness of the model in order to find out any Gender Bias

Fairness Check
"""
index = random.randrange(0, len(submission))

test_sample = df_test.ix[[index]]

test_sample2 = test_sample.copy()

# Change the Gender for test_sample2
test_sample2['CODE_GENDER_M'] = 1-test_sample2['CODE_GENDER_M'].iloc[0]
test_sample2['CODE_GENDER_F'] = 1-test_sample2['CODE_GENDER_F'].iloc[0]

print(test_sample[['CODE_GENDER_M','CODE_GENDER_F']])
print(test_sample2[['CODE_GENDER_M','CODE_GENDER_F']])

test_sample_pred = xgbModel.predict(scaler.transform(test_sample))[0]
test_sample2_pred = xgbModel.predict(scaler.transform(test_sample2))[0]

# Ideally these two should be equal
assert test_sample_pred == test_sample2_pred

# We could also check for drastic changes in prediction probabilities
# by putting a threshold for change in pred_prob
# print(xgbModel.predict_proba(scaler.transform(test_sample))
# print(xgbModel.predict_proba(scaler.transform(test_sample2)))

"""
Explainability: We could use LIME for generating explanations for individual instances
"""
np.random.seed(1)

explainer = LimeTabularExplainer(train_impute, class_names=[
                                 'pos', 'neg'], feature_names=features_impute, kernel_width=3, verbose=False)

# Choose a sample instance
instance_to_explain = test_impute[0]

exp = explainer.explain_instance(
    instance_to_explain, xgbModel.predict_proba, num_features=5)

assert len(exp.as_list()) == 5
print('Features responsible for prediction of instance_to_explain: ', exp.as_list())
exp.as_pyplot_figure()
