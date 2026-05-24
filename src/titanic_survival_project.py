

# Data handling library
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler

# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
import optuna
from preprocessing import preprocessing

preprocessing = preprocessing()
X, y = preprocessing.feature_engineering()
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

categorical_cols = ['Sex', 'Title', 'Embarked', 'Cabin']
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessing = ColumnTransformer(transformers=[
    ('categorical_pipeline', categorical_pipeline, categorical_cols)

], remainder='passthrough')

base_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('imputer', KNNImputer(missing_values=np.nan, n_neighbors=10, weights='uniform'))
    ])

logistic_model =  Pipeline(steps=[
    ('basepipeline', base_pipeline),
    ('scaling', RobustScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
logistic_model.fit(X_train, Y_train)


cv_sets = StratifiedKFold(n_splits=10, shuffle=True, random_state=56)
score = cross_val_score(logistic_model, X_train, Y_train, n_jobs=-1, scoring='accuracy', cv=cv_sets)
avg_log_score = np.mean(score)
print(f"The training score of logistic model is {score}\n")
print(f"The average training score of logistic model is {avg_log_score}\n")
log_predictions = logistic_model.predict(x_test)
print(f"The testing score accuracy  of logistic model is {accuracy_score(y_test, log_predictions)}\n")
classification_report_log = classification_report(y_test, log_predictions, output_dict=True)
precision_log = classification_report_log['1']['precision']
recall_log = classification_report_log['1']['recall']
accuracy_log = accuracy_score(y_test, log_predictions)
print(f'The precision and the recall of the Random Forest model is {precision_log} and {recall_log}')

#----RandomForest-----

def objective_RF(trail):
    n_estimators = trail.suggest_int('n_estimators', 100, 1000)
    min_samples_split = trail.suggest_int('min_samples_split', 2, 50)
    min_samples_leaf = trail.suggest_int('min_samples_leaf', 1, 40)
    max_depth = trail.suggest_int('max_depth', 5, 30)
    max_features = trail.suggest_categorical('max_features', ['sqrt', 'log2'])

    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_depth=max_depth, max_features=max_features)
    pipline = Pipeline(steps=[
        ('baseline', base_pipeline),
        ('model', model)
    ])
    cv_sets = StratifiedKFold(n_splits=10, shuffle=True, random_state=56)
    score = cross_val_score(estimator=pipline, X=X_train, y=Y_train, n_jobs=-1, cv=cv_sets)
    return score.mean()
study = optuna.create_study(direction='maximize')
study.optimize(objective_RF, n_trials=100, n_jobs=-1)


randomforest_pipeline = Pipeline(steps=[
    ('baseline', base_pipeline),
    ('rf_model', RandomForestClassifier(**study.best_params))
])
randomforest_pipeline.fit(X_train, Y_train)
print(f'The random forest model score on training data is {randomforest_pipeline.score(x_test, y_test)}')
RF_pred = randomforest_pipeline.predict(x_test)
print(f"The testing score accuracy  of Random Forest model is {accuracy_score(y_test, RF_pred)}\n")
classification_report_rf = classification_report(y_test, RF_pred, output_dict=True)
precision_rf = classification_report_rf['1']['precision']
recall_rf = classification_report_rf['1']['recall']
accuracy_rf = accuracy_score(y_test, RF_pred)
print(f'The precision and the recall of the Random Forest model is {precision_rf} and {recall_rf}')

#---------------XGBoost-----------------
def objective_xg(trails):
    n_estimators = trails.suggest_int("n_estimators", 100, 1000)
    max_depth = trails.suggest_int('max_depth', 5, 30)
    learning_rate = trails.suggest_float('learning_rate', 0.01, 0.1)
    subsample = trails.suggest_float('subsample', 0.7, 0.9)
    gamma = trails.suggest_float('gamma', 0, 5)

    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, gamma=gamma)
    pipeline = Pipeline(steps=[
        ('baseline', base_pipeline),
        ('model', model)
    ])
    cv_sets = StratifiedKFold(n_splits=10, shuffle=True, random_state=98)
    score = cross_val_score(estimator=pipeline, X=X_train, y=Y_train, n_jobs=-1, cv=cv_sets)
    return np.mean(score)
study_xg = optuna.create_study(direction='maximize')
study_xg.optimize(objective_xg, n_jobs=-1, n_trials=200)

xg_pipeline = Pipeline(steps=[
    ('baseline', base_pipeline),
    ('model', XGBClassifier(**study_xg.best_params))
])
xg_pipeline.fit(X_train, Y_train)



print(f'The random forest model score on training data is {xg_pipeline.score(x_test, y_test)}')
XG_pred = xg_pipeline.predict(x_test)
print(f"The testing score accuracy  of Random Forest model is {accuracy_score(y_test, XG_pred)}\n")
classification_report_xg = classification_report(y_test, XG_pred, output_dict=True)

precision_xg = classification_report_xg['1']['precision']
recall_xg = classification_report_xg['1']['recall']
accuracy_xg = accuracy_score(y_test, XG_pred)
print(f'The precision and the recall of the XGBoost model is {precision_xg} and {recall_xg}')

final_eval = {'Model':['Logistic Regression', 'Random Forest', 'XGBoost'],
              'Accuracy':[accuracy_log, accuracy_rf, accuracy_xg],
              'Precision':[precision_log, precision_rf, precision_xg],
              'Recall':[recall_log, recall_rf, recall_xg]}
final_result = pd.DataFrame(final_eval)
print(final_result)

