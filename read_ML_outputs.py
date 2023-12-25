from sklearn.metrics import accuracy_score, f1_score, r2_score, confusion_matrix, make_scorer, precision_score, recall_score
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from Preprocess4ML import *

file_path = 'finalized_model_PGV.sav'
with open(file_path, 'rb') as file:
    # Load the data using pickle
    clf_list = pickle.load(file)

dirname = os.path.dirname(__file__)
data_file = os.path.join(dirname, 'all_data_4_ML.h5')
data_input = pd.read_hdf(data_file, 'df')
ml_df = preprocessDataFrame(data_input)     # Prepare the df for ML
train, test = train_test_split(ml_df, test_size=0.20, random_state=42)

x_test = test.drop(labels = ["MIDR"], axis=1)
y_test = test['MIDR']
x_train = train.drop(labels = ["MIDR"], axis=1)
y_train = train['MIDR']

best_estimator = []
for clf_i in clf_list:
    clf_i.fit(x_train, y_train)
    best_estimator.append(clf_i.best_estimator_)

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


# ###
# import numpy as np
#
# file_path = 'saved_SVM_PGD.npy'
#
# # Load the variables from the .npy file
# loaded_data = np.load(file_path, allow_pickle=True)
#
# # Now you can access the variables
# f1_score1 = loaded_data[0]
# bias1 = loaded_data[1]
# variance1 = loaded_data[2]
# accuracy_score1 = loaded_data[3]
# precision_score1 = loaded_data[4]
# recall_score1 = loaded_data[5]


print('data')

