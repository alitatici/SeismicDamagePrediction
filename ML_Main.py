import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from collections import Counter
from Preprocess4ML import *
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, r2_score, confusion_matrix, make_scorer, precision_score, recall_score
import time
import pickle

start = time.time()
dirname = os.path.dirname(__file__)

# ##
# data_result_11 = pd.read_hdf(os.path.join(dirname, 'data_all_tables_parallel_0_540_rescaled_Ali.h5'), 'df')
# data_result_21 = pd.read_hdf(os.path.join(dirname, 'data_all_tables_parallel_540_1080_rescaled_Ali.h5'), 'df')
# data_result_31 = pd.read_hdf(os.path.join(dirname, 'data_all_tables_parallel_1080_1620_rescaled_Ali.h5'), 'df')
# data_result_41 = pd.read_hdf(os.path.join(dirname, 'data_all_tables_parallel_1620_2160_rescaled_Ali.h5'), 'df')
# data_result_51 = pd.read_hdf(os.path.join(dirname, 'data_all_tables_parallel_2160_2700_rescaled_Ali.h5'), 'df')
# data_result_61 = pd.read_hdf(os.path.join(dirname, 'data_all_tables_parallel_2700_3240_rescaled_Ali.h5'), 'df')
# data_result1 = pd.concat([data_result_11, data_result_21, data_result_31, data_result_41, data_result_51, data_result_61], axis=0)
# data_result1 = data_result1.reset_index(drop=True)
# maxMIDR = max(data_result1['MIDR'])
# data_result1['MIDR'] = data_result1['MIDR']. \
#     replace(data_result1.MIDR[data_result1.MIDR == -999].values, maxMIDR)
# data_result1.to_hdf(os.path.join(dirname, 'all_data_wo_failed_scale0p22.h5'), key='df', mode='w', format='table')
# data_result1.to_excel(os.path.join(dirname, 'all_data_wo_failed_scale0p22.xlsx'), index=False)
# ##

# ##
# data_file = os.path.join(dirname, 'completedDataset.h5')
# data_input = pd.read_hdf(data_file, 'df')
#
# data_file_failed = os.path.join(dirname, 'failedAnalyses_lastStep.h5')
# data_input_failed = pd.read_hdf(data_file_failed, 'df')
#
# all_data_wo_failed = mistakenlyFailed(data_input, data_input_failed)         # Prepare the df for ML
# all_data = collapsedRuns(all_data_wo_failed)
#
# all_data.to_hdf(os.path.join(dirname, 'all_data_wo_failed.h5'), key='df', mode='w', format='table')
# all_data.to_excel(os.path.join(dirname, 'all_data_wo_failed.xlsx'), index=False)
# ##

# ##
# data_file1 = os.path.join(dirname, 'all_data_wo_failed.h5')
# data_input1 = pd.read_hdf(data_file1, 'df')
# data_input1.insert(1, 'Scale Factor', 1.00)
#
# data_file2 = os.path.join(dirname, 'all_data_wo_failed_scale0p22.h5')
# data_input2 = pd.read_hdf(data_file2, 'df')
# data_input2.insert(1, 'Scale Factor', 0.22)
# data_input2['Run ID'] = data_input2['Run ID'] + 35640
#
# data_input = pd.concat([data_input1, data_input2], axis=0)
# data_input = data_input.reset_index(drop=True)
#
# data_input.to_hdf(os.path.join(dirname, 'all_data_4_ML.h5'), key='df', mode='w', format='table')
# data_input.to_excel(os.path.join(dirname, 'all_data_4_ML.xlsx'), index=False)
# ##

data_file = os.path.join(dirname, 'all_data_4_ML.h5')
data_input = pd.read_hdf(data_file, 'df')

ml_df = preprocessDataFrame(data_input)     # Prepare the df for ML
# visualizing(ml_df)             # Data visualizing

train, test = train_test_split(ml_df, test_size=0.20, random_state=42)

x_test = test.drop(labels = ["MIDR"], axis=1)
y_test = test['MIDR']
x_train = train.drop(labels = ["MIDR"], axis=1)
y_train = train['MIDR']

# decision tree, k-nearest neighbors, support vector machine, and random forest.
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
            SVC(random_state = random_state),
            RandomForestClassifier(random_state = random_state),
            KNeighborsClassifier()]

dt_parameter_grid = {"min_samples_split": range(10,500,20),
                    "max_depth": range(1,20,2)}

svc_param_grid = {"kernel": ["rbf"],
                  "gamma": [0.001, 0.01, 0.1, 1],
                  "C": [1, 10, 50, 100, 200, 300, 1000]}

rf_param_grid = {"max_features": [1, 3, 10],
                "min_samples_split": [2, 3, 10],
                "min_samples_leaf": [1, 3, 10],
                "bootstrap": [False],
                "n_estimators": [100, 300],
                "criterion": ["gini"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan"]}

classifier_param = [dt_parameter_grid,
                    svc_param_grid,
                    rf_param_grid,
                    knn_param_grid]

clf_list = []
cv_results = []
best_estimator = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], \
                        cv = StratifiedKFold(n_splits = 10), scoring = "f1_micro", n_jobs = -1, verbose = 1)
    clf.fit(x_train, y_train)
    cv_results.append(clf.best_score_)
    best_estimator.append(clf.best_estimator_)
    clf_list.append(clf)
    print(cv_results[i])

cv_result = pd.DataFrame({"Cross Validation Means": cv_results, "ML Models": ["DecisionTreeClassifier", "SVM", "RandomForest", "KNN"]})
# g = sns.barplot(x = "Cross Validation Means", y = "ML Models", data = cv_result)
# g.set_xlabel("Mean Accuracy")
# g.set_title("Cross Validation Scores")
# plt.show()

conf_mat0 = confusion_matrix(y_test, best_estimator[0].predict(x_test))
conf_mat1 = confusion_matrix(y_test, best_estimator[1].predict(x_test))
conf_mat2 = confusion_matrix(y_test, best_estimator[2].predict(x_test))
conf_mat3 = confusion_matrix(y_test, best_estimator[3].predict(x_test))

f1_score0 = f1_score(y_test, best_estimator[0].predict(x_test), average='macro')
f1_score1 = f1_score(y_test, best_estimator[1].predict(x_test), average='macro')
f1_score2 = f1_score(y_test, best_estimator[2].predict(x_test), average='macro')
f1_score3 = f1_score(y_test, best_estimator[3].predict(x_test), average='macro')

bias0 = np.round(np.mean((best_estimator[0].predict(x_test) - y_test.values) ** 2), 5)
bias1 = np.round(np.mean((best_estimator[1].predict(x_test) - y_test.values) ** 2), 5)
bias2 = np.round(np.mean((best_estimator[2].predict(x_test) - y_test.values) ** 2), 5)
bias3 = np.round(np.mean((best_estimator[3].predict(x_test) - y_test.values) ** 2), 5)

variance0 = np.round(np.var(best_estimator[0].predict(x_test)), 5)
variance1 = np.round(np.var(best_estimator[1].predict(x_test)), 5)
variance2 = np.round(np.var(best_estimator[2].predict(x_test)), 5)
variance3 = np.round(np.var(best_estimator[3].predict(x_test)), 5)

accuracy_score0 = accuracy_score(y_test, best_estimator[0].predict(x_test))
accuracy_score1 = accuracy_score(y_test, best_estimator[1].predict(x_test))
accuracy_score2 = accuracy_score(y_test, best_estimator[2].predict(x_test))
accuracy_score3 = accuracy_score(y_test, best_estimator[3].predict(x_test))

precision_score0 = precision_score(y_test, best_estimator[0].predict(x_test), average='macro')
precision_score1 = precision_score(y_test, best_estimator[1].predict(x_test), average='macro')
precision_score2 = precision_score(y_test, best_estimator[2].predict(x_test), average='macro')
precision_score3 = precision_score(y_test, best_estimator[3].predict(x_test), average='macro')

recall_score0 = recall_score(y_test, best_estimator[0].predict(x_test), average='macro')
recall_score1 = recall_score(y_test, best_estimator[1].predict(x_test), average='macro')
recall_score2 = recall_score(y_test, best_estimator[2].predict(x_test), average='macro')
recall_score3 = recall_score(y_test, best_estimator[3].predict(x_test), average='macro')

end = time.time()
time_spent = round((end - start), 2)
print(time_spent)

np.save('saved_DT_PGD.npy', [cv_result, f1_score0, bias0, variance0, accuracy_score0, precision_score0, recall_score0])
np.save('saved_mat_DT_PGD.npy', conf_mat0)
np.save('saved_SVM_PGD.npy', [f1_score1, bias1, variance1, accuracy_score1, precision_score1, recall_score1])
np.save('saved_mat_SVM_PGD.npy', conf_mat1)
np.save('saved_RF_PGD.npy', [f1_score2, bias2, variance2, accuracy_score2, precision_score2, recall_score2])
np.save('saved_mat_RF_PGD.npy', conf_mat2)
np.save('saved_KNN_PGD.npy', [f1_score3, bias3, variance3, accuracy_score3, precision_score3, recall_score3])
np.save('saved_mat_KNN_PGD.npy', conf_mat3)
np.save('saved_runPGD_cv_result.npy', cv_result)
np.save('saved_runPGD_best_estimator.npy', best_estimator)
np.save('saved_runPGD_clf.npy', clf)

filename = 'finalized_model_' + 'PGD' + '.sav'
model_folder = os.path.join(dirname, filename)
pickle.dump(clf_list, open(model_folder, 'wb'))

print('debug')

